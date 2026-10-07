"""Persist per-job provider/model selections."""
from alembic import op
import sqlalchemy as sa

revision = "0004_job_options"
down_revision = "0003_token_usage"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column("jobs", sa.Column("options", sa.JSON(), nullable=True))

def downgrade() -> None:
    op.drop_column("jobs", "options")
