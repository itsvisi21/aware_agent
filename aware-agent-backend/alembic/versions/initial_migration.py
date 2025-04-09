"""initial migration

Revision ID: initial
Revises: 
Create Date: 2023-01-01 00:00:00.000000

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = 'initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('messages', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create agents table
    op.create_table(
        'agents',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create metrics table
    op.create_table(
        'metrics',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('labels', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_conversations_created_at', 'conversations', ['created_at'])
    op.create_index('ix_agents_type', 'agents', ['type'])
    op.create_index('ix_metrics_timestamp', 'metrics', ['timestamp'])
    op.create_index('ix_metrics_name', 'metrics', ['name'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_metrics_name')
    op.drop_index('ix_metrics_timestamp')
    op.drop_index('ix_agents_type')
    op.drop_index('ix_conversations_created_at')

    # Drop tables
    op.drop_table('metrics')
    op.drop_table('agents')
    op.drop_table('conversations')
