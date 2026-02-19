-- migrate:up
CREATE TABLE IF NOT EXISTS model_aliases (
    user_id VARCHAR NOT NULL,
    alias VARCHAR(64) NOT NULL,
    chute_ids JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, alias),
    CONSTRAINT alias_ascii_no_colon CHECK (alias ~ '^[\x21-\x39\x3B-\x7E]+$')
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_aliases_user_alias_lower ON model_aliases (user_id, LOWER(alias));

-- migrate:down
DROP TABLE IF EXISTS model_aliases;
