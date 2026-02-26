"""Initial migration."""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(50), unique=True, nullable=False),
        sa.Column('email', sa.String(100), unique=True, nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Create user_progress table
    op.create_table(
        'user_progress',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('lesson_id', sa.Integer(), nullable=False),
        sa.Column('xp_gained', sa.Integer(), default=0),
        sa.Column('completed_at', sa.DateTime(), nullable=False),
        sa.Column('current_level', sa.Integer(), default=1),
        sa.Column('total_xp', sa.Integer(), default=0),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_user_progress_user_id'), 'user_progress', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_progress_lesson_id'), 'user_progress', ['lesson_id'], unique=False)

    # Create streaks table
    op.create_table(
        'streaks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), unique=True, nullable=False),
        sa.Column('current_streak', sa.Integer(), default=0),
        sa.Column('longest_streak', sa.Integer(), default=0),
        sa.Column('last_activity_date', sa.DateTime(), nullable=False),
        sa.Column('is_active_today', sa.Boolean(), default=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_streaks_user_id'), 'streaks', ['user_id'], unique=True)

    # Create game_sessions table
    op.create_table(
        'game_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('lesson_id', sa.Integer(), nullable=False),
        sa.Column('score', sa.Integer(), default=0),
        sa.Column('accuracy', sa.Float(), default=0.0),
        sa.Column('duration_seconds', sa.Integer(), default=0),
        sa.Column('completed', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_game_sessions_user_id'), 'game_sessions', ['user_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_game_sessions_user_id'), table_name='game_sessions')
    op.drop_table('game_sessions')
    op.drop_index(op.f('ix_streaks_user_id'), table_name='streaks')
    op.drop_table('streaks')
    op.drop_index(op.f('ix_user_progress_lesson_id'), table_name='user_progress')
    op.drop_index(op.f('ix_user_progress_user_id'), table_name='user_progress')
    op.drop_table('user_progress')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_table('users')
