from sqlalchemy.orm import Session

def get_game_titles(db: Session, limit: int = 10):
    # TODO: implémentation réelle (SELECT ... FROM games)
    return ["Sample Game A", "Sample Game B"][:limit]
