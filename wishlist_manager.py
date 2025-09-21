# wishlist_manager.py - Gestionnaire de wishlist avec notifications
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pymysql

logger = logging.getLogger("wishlist-manager")

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    READ = "read"

@dataclass
class WishlistItem:
    """Item de wishlist"""
    id: Optional[int]
    user_id: int
    game_title: str
    max_price: float
    currency: str = "EUR"
    created_at: Optional[datetime] = None
    is_active: bool = True
    notification_count: int = 0
    last_notification: Optional[datetime] = None

@dataclass
class PriceAlert:
    """Alerte de prix"""
    id: Optional[int]
    wishlist_item_id: int
    game_title: str
    current_price: float
    threshold_price: float
    shop_name: str
    shop_url: str
    created_at: Optional[datetime] = None
    status: NotificationStatus = NotificationStatus.PENDING
    user_id: int = None

class WishlistManager:
    """Gestionnaire de wishlist avec notifications de prix"""
    
    def __init__(self, db_config: Dict[str, any]):
        self.db_config = db_config
        self._ensure_tables()
    
    def _get_connection(self):
        """Obtient une connexion à la base de données"""
        return pymysql.connect(
            host=self.db_config.get("host", "localhost"),
            port=self.db_config.get("port", 3306),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            database=self.db_config.get("database"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
    
    def _ensure_tables(self):
        """Crée les tables nécessaires pour la wishlist"""
        create_wishlist_table = """
        CREATE TABLE IF NOT EXISTS user_wishlist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            game_title VARCHAR(500) NOT NULL,
            max_price DECIMAL(10,2) NOT NULL,
            currency VARCHAR(3) DEFAULT 'EUR',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            notification_count INT DEFAULT 0,
            last_notification TIMESTAMP NULL,
            
            INDEX idx_user_id (user_id),
            INDEX idx_game_title (game_title(100)),
            INDEX idx_is_active (is_active),
            UNIQUE KEY unique_user_game (user_id, game_title)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        create_price_alerts_table = """
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            wishlist_item_id INT NOT NULL,
            user_id INT NOT NULL,
            game_title VARCHAR(500) NOT NULL,
            current_price DECIMAL(10,2) NOT NULL,
            threshold_price DECIMAL(10,2) NOT NULL,
            shop_name VARCHAR(255) NOT NULL,
            shop_url TEXT,
            currency VARCHAR(3) DEFAULT 'EUR',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status ENUM('pending', 'sent', 'read') DEFAULT 'pending',
            
            FOREIGN KEY (wishlist_item_id) REFERENCES user_wishlist(id) ON DELETE CASCADE,
            INDEX idx_user_id (user_id),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        create_notification_settings_table = """
        CREATE TABLE IF NOT EXISTS notification_settings (
            user_id INT PRIMARY KEY,
            email_notifications BOOLEAN DEFAULT TRUE,
            push_notifications BOOLEAN DEFAULT TRUE,
            notification_frequency ENUM('immediate', 'daily', 'weekly') DEFAULT 'immediate',
            quiet_hours_start TIME DEFAULT '22:00:00',
            quiet_hours_end TIME DEFAULT '08:00:00',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_wishlist_table)
                    cursor.execute(create_price_alerts_table)
                    cursor.execute(create_notification_settings_table)
                    
            logger.info("Wishlist tables created/verified successfully")
            
        except Exception as e:
            logger.error(f"Failed to create wishlist tables: {e}")
            raise
    
    def add_to_wishlist(self, user_id: int, game_title: str, max_price: float, currency: str = "EUR") -> bool:
        """Ajoute un jeu à la wishlist"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Vérifier si le jeu existe déjà dans la wishlist
                    cursor.execute(
                        "SELECT id FROM user_wishlist WHERE user_id = %s AND game_title = %s",
                        (user_id, game_title)
                    )
                    
                    if cursor.fetchone():
                        # Mettre à jour le prix maximum
                        cursor.execute("""
                            UPDATE user_wishlist 
                            SET max_price = %s, currency = %s, updated_at = CURRENT_TIMESTAMP, is_active = TRUE
                            WHERE user_id = %s AND game_title = %s
                        """, (max_price, currency, user_id, game_title))
                        
                        logger.info(f"Updated wishlist item for user {user_id}: {game_title} <= {max_price} {currency}")
                    else:
                        # Ajouter nouveau item
                        cursor.execute("""
                            INSERT INTO user_wishlist (user_id, game_title, max_price, currency)
                            VALUES (%s, %s, %s, %s)
                        """, (user_id, game_title, max_price, currency))
                        
                        logger.info(f"Added to wishlist for user {user_id}: {game_title} <= {max_price} {currency}")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to add to wishlist: {e}")
            return False
    
    def remove_from_wishlist(self, user_id: int, wishlist_id: int) -> bool:
        """Supprime un item de la wishlist"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM user_wishlist WHERE id = %s AND user_id = %s",
                        (wishlist_id, user_id)
                    )
                    
                    return cursor.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Failed to remove from wishlist: {e}")
            return False
    
    def get_user_wishlist(self, user_id: int) -> List[WishlistItem]:
        """Récupère la wishlist d'un utilisateur"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, user_id, game_title, max_price, currency, created_at, 
                               is_active, notification_count, last_notification
                        FROM user_wishlist 
                        WHERE user_id = %s AND is_active = TRUE
                        ORDER BY created_at DESC
                    """, (user_id,))
                    
                    items = []
                    for row in cursor.fetchall():
                        items.append(WishlistItem(
                            id=row['id'],
                            user_id=row['user_id'],
                            game_title=row['game_title'],
                            max_price=float(row['max_price']),
                            currency=row['currency'],
                            created_at=row['created_at'],
                            is_active=row['is_active'],
                            notification_count=row['notification_count'],
                            last_notification=row['last_notification']
                        ))
                    
                    return items
                    
        except Exception as e:
            logger.error(f"Failed to get wishlist: {e}")
            return []
    
    def check_price_alerts(self) -> List[PriceAlert]:
        """Vérifie les alertes de prix en comparant avec best_price_pc"""
        alerts = []
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Requête pour trouver les jeux en wishlist avec des prix inférieurs au seuil
                    query = """
                    SELECT 
                        w.id as wishlist_id,
                        w.user_id,
                        w.game_title,
                        w.max_price,
                        w.currency,
                        p.best_price_PC,
                        p.best_shop_PC,
                        p.site_url_PC,
                        p.title as exact_title
                    FROM user_wishlist w
                    LEFT JOIN best_price_pc p ON (
                        LOWER(TRIM(p.title)) LIKE CONCAT('%', LOWER(TRIM(w.game_title)), '%')
                        OR LOWER(TRIM(w.game_title)) LIKE CONCAT('%', LOWER(TRIM(p.title)), '%')
                    )
                    WHERE w.is_active = TRUE 
                    AND p.best_price_PC IS NOT NULL
                    AND p.best_price_PC != ''
                    """
                    
                    cursor.execute(query)
                    matches = cursor.fetchall()
                    
                    for match in matches:
                        try:
                            # Parser le prix (gérer différents formats)
                            price_str = match['best_price_PC']
                            current_price = self._parse_price(price_str)
                            
                            if current_price is not None and current_price <= float(match['max_price']):
                                # Vérifier si on n'a pas déjà envoyé une alerte récemment
                                if not self._has_recent_alert(match['wishlist_id'], current_price):
                                    alert = PriceAlert(
                                        id=None,
                                        wishlist_item_id=match['wishlist_id'],
                                        user_id=match['user_id'],
                                        game_title=match['exact_title'] or match['game_title'],
                                        current_price=current_price,
                                        threshold_price=float(match['max_price']),
                                        shop_name=match['best_shop_PC'] or "Unknown",
                                        shop_url=match['site_url_PC'] or "",
                                        created_at=datetime.utcnow()
                                    )
                                    
                                    # Sauvegarder l'alerte
                                    alert_id = self._save_alert(alert)
                                    if alert_id:
                                        alert.id = alert_id
                                        alerts.append(alert)
                                        
                                        # Mettre à jour les stats de notification
                                        self._update_notification_stats(match['wishlist_id'])
                        
                        except Exception as e:
                            logger.warning(f"Error processing price match: {e}")
                            continue
                    
        except Exception as e:
            logger.error(f"Failed to check price alerts: {e}")
        
        return alerts
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse un prix depuis différents formats"""
        if not price_str:
            return None
        
        import re
        
        # Nettoyer la chaîne
        clean_price = re.sub(r'[^\d.,€$]', '', price_str)
        
        # Patterns de prix courants
        patterns = [
            r'(\d+[.,]\d+)',  # 19.99 ou 19,99
            r'(\d+)',         # 19
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_price)
            if match:
                try:
                    price_value = float(match.group(1).replace(',', '.'))
                    return price_value
                except ValueError:
                    continue
        
        return None
    
    def _has_recent_alert(self, wishlist_item_id: int, price: float, hours: int = 24) -> bool:
        """Vérifie si une alerte récente existe pour ce prix"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) as count
                        FROM price_alerts 
                        WHERE wishlist_item_id = %s 
                        AND ABS(current_price - %s) < 0.50
                        AND created_at > DATE_SUB(NOW(), INTERVAL %s HOUR)
                    """, (wishlist_item_id, price, hours))
                    
                    result = cursor.fetchone()
                    return result['count'] > 0
                    
        except Exception as e:
            logger.error(f"Error checking recent alerts: {e}")
            return False
    
    def _save_alert(self, alert: PriceAlert) -> Optional[int]:
        """Sauvegarde une alerte en base"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO price_alerts 
                        (wishlist_item_id, user_id, game_title, current_price, threshold_price, 
                         shop_name, shop_url, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        alert.wishlist_item_id,
                        alert.user_id,
                        alert.game_title,
                        alert.current_price,
                        alert.threshold_price,
                        alert.shop_name,
                        alert.shop_url,
                        alert.status.value
                    ))
                    
                    return cursor.lastrowid
                    
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return None
    
    def _update_notification_stats(self, wishlist_item_id: int):
        """Met à jour les statistiques de notification"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE user_wishlist 
                        SET notification_count = notification_count + 1,
                            last_notification = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (wishlist_item_id,))
                    
        except Exception as e:
            logger.error(f"Failed to update notification stats: {e}")
    
    def get_user_alerts(self, user_id: int, status: Optional[NotificationStatus] = None) -> List[PriceAlert]:
        """Récupère les alertes d'un utilisateur"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    base_query = """
                        SELECT id, wishlist_item_id, user_id, game_title, current_price,
                               threshold_price, shop_name, shop_url, created_at, status
                        FROM price_alerts 
                        WHERE user_id = %s
                    """
                    
                    params = [user_id]
                    
                    if status:
                        base_query += " AND status = %s"
                        params.append(status.value)
                    
                    base_query += " ORDER BY created_at DESC LIMIT 50"
                    
                    cursor.execute(base_query, params)
                    
                    alerts = []
                    for row in cursor.fetchall():
                        alerts.append(PriceAlert(
                            id=row['id'],
                            wishlist_item_id=row['wishlist_item_id'],
                            user_id=row['user_id'],
                            game_title=row['game_title'],
                            current_price=float(row['current_price']),
                            threshold_price=float(row['threshold_price']),
                            shop_name=row['shop_name'],
                            shop_url=row['shop_url'],
                            created_at=row['created_at'],
                            status=NotificationStatus(row['status'])
                        ))
                    
                    return alerts
                    
        except Exception as e:
            logger.error(f"Failed to get user alerts: {e}")
            return []
    
    def mark_alert_as_read(self, user_id: int, alert_id: int) -> bool:
        """Marque une alerte comme lue"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE price_alerts 
                        SET status = 'read'
                        WHERE id = %s AND user_id = %s
                    """, (alert_id, user_id))
                    
                    return cursor.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Failed to mark alert as read: {e}")
            return False
    
    def get_wishlist_stats(self, user_id: int) -> Dict[str, any]:
        """Statistiques de la wishlist utilisateur"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Stats wishlist
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_items,
                            AVG(max_price) as avg_price_threshold,
                            SUM(notification_count) as total_notifications
                        FROM user_wishlist 
                        WHERE user_id = %s AND is_active = TRUE
                    """, (user_id,))
                    
                    wishlist_stats = cursor.fetchone()
                    
                    # Stats alertes
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_alerts,
                            COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_alerts,
                            COUNT(CASE WHEN status = 'read' THEN 1 END) as read_alerts,
                            AVG(current_price - threshold_price) as avg_savings
                        FROM price_alerts 
                        WHERE user_id = %s
                        AND created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                    """, (user_id,))
                    
                    alert_stats = cursor.fetchone()
                    
                    return {
                        "wishlist": {
                            "total_items": wishlist_stats['total_items'] or 0,
                            "avg_price_threshold": float(wishlist_stats['avg_price_threshold'] or 0),
                            "total_notifications": wishlist_stats['total_notifications'] or 0
                        },
                        "alerts_30_days": {
                            "total_alerts": alert_stats['total_alerts'] or 0,
                            "pending_alerts": alert_stats['pending_alerts'] or 0,
                            "read_alerts": alert_stats['read_alerts'] or 0,
                            "avg_savings": float(alert_stats['avg_savings'] or 0)
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get wishlist stats: {e}")
            return {"wishlist": {}, "alerts_30_days": {}}


# Singleton pour l'accès global
_wishlist_manager_instance: Optional[WishlistManager] = None

def get_wishlist_manager() -> Optional[WishlistManager]:
    """Récupère l'instance du gestionnaire de wishlist"""
    global _wishlist_manager_instance
    
    if _wishlist_manager_instance is None:
        try:
            from settings import get_settings
            settings = get_settings()
            
            if settings.db_configured:
                db_config = {
                    "host": settings.DB_HOST,
                    "port": settings.DB_PORT,
                    "user": settings.DB_USER,
                    "password": settings.DB_PASSWORD,
                    "database": settings.DB_NAME
                }
                
                _wishlist_manager_instance = WishlistManager(db_config)
                logger.info("Wishlist manager initialized successfully")
            else:
                logger.warning("Database not configured - wishlist features disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize wishlist manager: {e}")
    
    return _wishlist_manager_instance

def reset_wishlist_manager():
    """Reset l'instance du gestionnaire"""
    global _wishlist_manager_instance
    _wishlist_manager_instance = None
