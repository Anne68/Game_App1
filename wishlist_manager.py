# wishlist_manager.py - Gestionnaire de wishlist avec notifications
from __future__ import annotations

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import pymysql
from pydantic import BaseModel, Field

logger = logging.getLogger("wishlist-manager")

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"

class NotificationType(Enum):
    PRICE_DROP = "price_drop"
    AVAILABILITY = "availability"
    WISHLIST_REMINDER = "wishlist_reminder"

@dataclass
class WishlistItem:
    """Item de wishlist"""
    id: Optional[int] = None
    user_id: int = 0
    game_title: str = ""
    target_price: float = 0.0
    current_price: Optional[float] = None
    price_currency: str = "EUR"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    notification_sent: bool = False

@dataclass
class PriceAlert:
    """Alerte de prix"""
    id: Optional[int] = None
    wishlist_item_id: int = 0
    old_price: Optional[float] = None
    new_price: float = 0.0
    shop_name: str = ""
    notification_type: NotificationType = NotificationType.PRICE_DROP
    notification_status: NotificationStatus = NotificationStatus.PENDING
    created_at: Optional[datetime] = None
    message: str = ""

# === Models Pydantic pour l'API ===

class WishlistAddRequest(BaseModel):
    game_title: str = Field(..., min_length=1, max_length=255)
    target_price: float = Field(..., gt=0, le=1000)
    price_currency: str = Field(default="EUR", pattern="^(EUR|USD|GBP)$")

class WishlistUpdateRequest(BaseModel):
    target_price: Optional[float] = Field(None, gt=0, le=1000)
    is_active: Optional[bool] = None

class WishlistResponse(BaseModel):
    id: int
    game_title: str
    target_price: float
    current_price: Optional[float]
    price_currency: str
    created_at: str
    is_active: bool
    notification_sent: bool
    price_difference: Optional[float] = None
    alert_triggered: bool = False

class NotificationResponse(BaseModel):
    id: int
    wishlist_item_id: int
    message: str
    notification_type: str
    created_at: str
    is_read: bool = False

class WishlistManager:
    """Gestionnaire principal de la wishlist"""
    
    def __init__(self, db_connection_func):
        self.get_db_conn = db_connection_func
        self.price_check_interval = 3600  # 1 heure
        self.notification_batch_size = 50
        
    # === Gestion de la base de données ===
    
    def ensure_wishlist_tables(self):
        """Crée les tables de wishlist si elles n'existent pas"""
        
        create_tables_sql = """
        -- Table wishlist
        CREATE TABLE IF NOT EXISTS user_wishlist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            game_title VARCHAR(500) NOT NULL,
            target_price DECIMAL(10,2) NOT NULL,
            current_price DECIMAL(10,2) NULL,
            price_currency VARCHAR(3) DEFAULT 'EUR',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            notification_sent BOOLEAN DEFAULT FALSE,
            
            INDEX idx_user_id (user_id),
            INDEX idx_game_title (game_title),
            INDEX idx_is_active (is_active),
            INDEX idx_target_price (target_price)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        
        -- Table des alertes de prix
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            wishlist_item_id INT NOT NULL,
            old_price DECIMAL(10,2) NULL,
            new_price DECIMAL(10,2) NOT NULL,
            shop_name VARCHAR(255) DEFAULT '',
            notification_type ENUM('price_drop', 'availability', 'wishlist_reminder') DEFAULT 'price_drop',
            notification_status ENUM('pending', 'sent', 'failed') DEFAULT 'pending',
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read BOOLEAN DEFAULT FALSE,
            
            FOREIGN KEY (wishlist_item_id) REFERENCES user_wishlist(id) ON DELETE CASCADE,
            INDEX idx_wishlist_item (wishlist_item_id),
            INDEX idx_notification_status (notification_status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        
        -- Table des logs de notifications
        CREATE TABLE IF NOT EXISTS notification_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            wishlist_item_id INT NOT NULL,
            notification_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            message TEXT,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_user_id (user_id),
            INDEX idx_sent_at (sent_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                # Exécuter chaque instruction CREATE TABLE séparément
                for statement in create_tables_sql.split(';'):
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        cursor.execute(statement)
            conn.commit()
        
        logger.info("Wishlist tables ensured")
    
    # === CRUD Operations ===
    
    def add_to_wishlist(self, user_id: int, request: WishlistAddRequest) -> WishlistItem:
        """Ajoute un jeu à la wishlist"""
        
        # Vérifier si le jeu n'est pas déjà dans la wishlist
        existing = self.get_wishlist_item_by_title(user_id, request.game_title)
        if existing:
            raise ValueError(f"Game '{request.game_title}' already in wishlist")
        
        # Chercher le prix actuel du jeu
        current_price = self._find_current_price(request.game_title)
        
        sql = """
        INSERT INTO user_wishlist (user_id, game_title, target_price, current_price, price_currency)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    user_id, 
                    request.game_title, 
                    request.target_price, 
                    current_price,
                    request.price_currency
                ))
                wishlist_id = cursor.lastrowid
            conn.commit()
        
        # Vérifier immédiatement si le prix cible est déjà atteint
        if current_price and current_price <= request.target_price:
            self._create_price_alert(wishlist_id, None, current_price, "Price target already met!")
        
        logger.info(f"Added '{request.game_title}' to wishlist for user {user_id}")
        
        return self.get_wishlist_item(wishlist_id)
    
    def get_user_wishlist(self, user_id: int, active_only: bool = True) -> List[WishlistItem]:
        """Récupère la wishlist d'un utilisateur"""
        
        sql = """
        SELECT id, user_id, game_title, target_price, current_price, price_currency,
               created_at, updated_at, is_active, notification_sent
        FROM user_wishlist 
        WHERE user_id = %s
        """
        
        params = [user_id]
        
        if active_only:
            sql += " AND is_active = 1"
        
        sql += " ORDER BY created_at DESC"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        
        wishlist_items = []
        for row in rows:
            item = WishlistItem(
                id=row['id'],
                user_id=row['user_id'],
                game_title=row['game_title'],
                target_price=float(row['target_price']),
                current_price=float(row['current_price']) if row['current_price'] else None,
                price_currency=row['price_currency'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                is_active=bool(row['is_active']),
                notification_sent=bool(row['notification_sent'])
            )
            wishlist_items.append(item)
        
        return wishlist_items
    
    def get_wishlist_item(self, wishlist_id: int) -> Optional[WishlistItem]:
        """Récupère un item de wishlist par ID"""
        
        sql = """
        SELECT id, user_id, game_title, target_price, current_price, price_currency,
               created_at, updated_at, is_active, notification_sent
        FROM user_wishlist 
        WHERE id = %s
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (wishlist_id,))
                row = cursor.fetchone()
        
        if not row:
            return None
        
        return WishlistItem(
            id=row['id'],
            user_id=row['user_id'],
            game_title=row['game_title'],
            target_price=float(row['target_price']),
            current_price=float(row['current_price']) if row['current_price'] else None,
            price_currency=row['price_currency'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            is_active=bool(row['is_active']),
            notification_sent=bool(row['notification_sent'])
        )
    
    def get_wishlist_item_by_title(self, user_id: int, game_title: str) -> Optional[WishlistItem]:
        """Récupère un item de wishlist par titre"""
        
        sql = """
        SELECT id, user_id, game_title, target_price, current_price, price_currency,
               created_at, updated_at, is_active, notification_sent
        FROM user_wishlist 
        WHERE user_id = %s AND game_title = %s
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (user_id, game_title))
                row = cursor.fetchone()
        
        if not row:
            return None
        
        return WishlistItem(
            id=row['id'],
            user_id=row['user_id'],
            game_title=row['game_title'],
            target_price=float(row['target_price']),
            current_price=float(row['current_price']) if row['current_price'] else None,
            price_currency=row['price_currency'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            is_active=bool(row['is_active']),
            notification_sent=bool(row['notification_sent'])
        )
    
    def update_wishlist_item(self, wishlist_id: int, user_id: int, request: WishlistUpdateRequest) -> WishlistItem:
        """Met à jour un item de wishlist"""
        
        # Vérifier que l'item appartient à l'utilisateur
        item = self.get_wishlist_item(wishlist_id)
        if not item or item.user_id != user_id:
            raise ValueError("Wishlist item not found or access denied")
        
        update_parts = []
        params = []
        
        if request.target_price is not None:
            update_parts.append("target_price = %s")
            params.append(request.target_price)
        
        if request.is_active is not None:
            update_parts.append("is_active = %s")
            params.append(request.is_active)
        
        if not update_parts:
            return item
        
        update_parts.append("updated_at = CURRENT_TIMESTAMP")
        params.append(wishlist_id)
        
        sql = f"UPDATE user_wishlist SET {', '.join(update_parts)} WHERE id = %s"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
            conn.commit()
        
        logger.info(f"Updated wishlist item {wishlist_id}")
        
        return self.get_wishlist_item(wishlist_id)
    
    def remove_from_wishlist(self, wishlist_id: int, user_id: int) -> bool:
        """Supprime un item de la wishlist"""
        
        # Vérifier que l'item appartient à l'utilisateur
        item = self.get_wishlist_item(wishlist_id)
        if not item or item.user_id != user_id:
            raise ValueError("Wishlist item not found or access denied")
        
        sql = "DELETE FROM user_wishlist WHERE id = %s AND user_id = %s"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (wishlist_id, user_id))
                deleted = cursor.rowcount > 0
            conn.commit()
        
        if deleted:
            logger.info(f"Removed wishlist item {wishlist_id} for user {user_id}")
        
        return deleted
    
    # === Gestion des prix et notifications ===
    
    def _find_current_price(self, game_title: str) -> Optional[float]:
        """Trouve le prix actuel d'un jeu dans la table best_price_pc"""
        
        sql = """
        SELECT best_price_PC, similarity_score 
        FROM best_price_pc 
        WHERE LOWER(title) LIKE LOWER(%s) 
        ORDER BY similarity_score DESC 
        LIMIT 1
        """
        
        try:
            with self.get_db_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (f"%{game_title}%",))
                    row = cursor.fetchone()
            
            if row and row['best_price_PC']:
                price_str = row['best_price_PC']
                # Extraire le prix de chaînes comme "€12.99" ou "$19.99"
                import re
                price_match = re.search(r'[\d,]+\.?\d*', price_str.replace(',', '.'))
                if price_match:
                    return float(price_match.group())
        
        except Exception as e:
            logger.warning(f"Error finding current price for '{game_title}': {e}")
        
        return None
    
    def check_price_alerts(self) -> int:
        """Vérifie tous les items de wishlist pour les alertes de prix"""
        
        logger.info("Starting price alerts check...")
        
        # Récupérer tous les items actifs de wishlist
        sql = """
        SELECT id, user_id, game_title, target_price, current_price
        FROM user_wishlist 
        WHERE is_active = 1 AND notification_sent = 0
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                items = cursor.fetchall()
        
        alerts_created = 0
        
        for item in items:
            try:
                # Chercher le nouveau prix
                new_price = self._find_current_price(item['game_title'])
                
                if new_price is None:
                    continue
                
                old_price = float(item['current_price']) if item['current_price'] else None
                target_price = float(item['target_price'])
                
                # Vérifier si le prix cible est atteint
                if new_price <= target_price:
                    message = f"🎯 Price target reached for '{item['game_title']}'! Current price: €{new_price:.2f} (target: €{target_price:.2f})"
                    
                    self._create_price_alert(
                        item['id'], old_price, new_price, message,
                        NotificationType.PRICE_DROP
                    )
                    
                    # Marquer comme notifié
                    self._mark_notification_sent(item['id'])
                    
                    alerts_created += 1
                    logger.info(f"Price alert created for '{item['game_title']}'")
                
                # Mettre à jour le prix actuel même si pas d'alerte
                elif old_price != new_price:
                    self._update_current_price(item['id'], new_price)
            
            except Exception as e:
                logger.error(f"Error checking price for item {item['id']}: {e}")
        
        logger.info(f"Price check completed. Created {alerts_created} alerts.")
        return alerts_created
    
    def _create_price_alert(self, wishlist_item_id: int, old_price: Optional[float], 
                           new_price: float, message: str, 
                           notification_type: NotificationType = NotificationType.PRICE_DROP):
        """Crée une alerte de prix"""
        
        sql = """
        INSERT INTO price_alerts (wishlist_item_id, old_price, new_price, message, notification_type)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (
                    wishlist_item_id, old_price, new_price, message, notification_type.value
                ))
            conn.commit()
    
    def _mark_notification_sent(self, wishlist_item_id: int):
        """Marque un item comme notifié"""
        
        sql = "UPDATE user_wishlist SET notification_sent = 1 WHERE id = %s"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (wishlist_item_id,))
            conn.commit()
    
    def _update_current_price(self, wishlist_item_id: int, new_price: float):
        """Met à jour le prix actuel d'un item"""
        
        sql = "UPDATE user_wishlist SET current_price = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (new_price, wishlist_item_id))
            conn.commit()
    
    # === Notifications ===
    
    def get_user_notifications(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère les notifications d'un utilisateur"""
        
        sql = """
        SELECT pa.id, pa.wishlist_item_id, pa.message, pa.notification_type, 
               pa.created_at, pa.is_read, uw.game_title, pa.new_price
        FROM price_alerts pa
        JOIN user_wishlist uw ON pa.wishlist_item_id = uw.id
        WHERE uw.user_id = %s
        ORDER BY pa.created_at DESC
        LIMIT %s
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (user_id, limit))
                rows = cursor.fetchall()
        
        notifications = []
        for row in rows:
            notifications.append({
                "id": row['id'],
                "wishlist_item_id": row['wishlist_item_id'],
                "game_title": row['game_title'],
                "message": row['message'],
                "notification_type": row['notification_type'],
                "price": float(row['new_price']),
                "created_at": row['created_at'].isoformat(),
                "is_read": bool(row['is_read'])
            })
        
        return notifications
    
    def mark_notification_read(self, notification_id: int, user_id: int) -> bool:
        """Marque une notification comme lue"""
        
        sql = """
        UPDATE price_alerts pa
        JOIN user_wishlist uw ON pa.wishlist_item_id = uw.id
        SET pa.is_read = 1
        WHERE pa.id = %s AND uw.user_id = %s
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (notification_id, user_id))
                updated = cursor.rowcount > 0
            conn.commit()
        
        return updated
    
    def get_notification_count(self, user_id: int, unread_only: bool = True) -> int:
        """Compte les notifications d'un utilisateur"""
        
        sql = """
        SELECT COUNT(*) as count
        FROM price_alerts pa
        JOIN user_wishlist uw ON pa.wishlist_item_id = uw.id
        WHERE uw.user_id = %s
        """
        
        params = [user_id]
        
        if unread_only:
            sql += " AND pa.is_read = 0"
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                row = cursor.fetchone()
        
        return row['count'] if row else 0
    
    # === Tâches de maintenance ===
    
    async def start_price_monitoring(self):
        """Démarre la surveillance des prix en arrière-plan"""
        logger.info("Starting price monitoring service...")
        
        while True:
            try:
                await asyncio.sleep(self.price_check_interval)
                self.check_price_alerts()
            except Exception as e:
                logger.error(f"Error in price monitoring: {e}")
                await asyncio.sleep(60)  # Attendre 1 minute en cas d'erreur
    
    def cleanup_old_notifications(self, days_old: int = 30) -> int:
        """Nettoie les anciennes notifications"""
        
        sql = """
        DELETE FROM price_alerts 
        WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        
        with self.get_db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (days_old,))
                deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted} old notifications")
        return deleted
