-- Enable Row Level Security
ALTER TABLE models ENABLE ROW LEVEL SECURITY;

-- Create a secure users table with proper constraints
CREATE TABLE secure_users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    failed_attempts INTEGER DEFAULT 0,
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Create secure models table
CREATE TABLE secure_models (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    owner_id INTEGER REFERENCES secure_users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    is_public BOOLEAN DEFAULT false,
    CONSTRAINT valid_name CHECK (name ~ '^[A-Za-z0-9\s\-_]{1,100}$')
);

-- Create RLS policies
CREATE POLICY model_access_policy ON secure_models
    FOR ALL
    USING (
        owner_id = CURRENT_USER_ID() 
        OR is_public = true
        OR EXISTS (
            SELECT 1 FROM secure_users 
            WHERE user_id = CURRENT_USER_ID() 
            AND is_active = true
        )
    );

-- Create function to rate limit login attempts
CREATE OR REPLACE FUNCTION check_login_attempts()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.failed_attempts >= 5 THEN
        NEW.is_active := false;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for login attempts
CREATE TRIGGER login_attempt_check
    BEFORE UPDATE ON secure_users
    FOR EACH ROW
    EXECUTE FUNCTION check_login_attempts();

-- Add additional security measures
ALTER TABLE secure_users ADD CONSTRAINT password_strength 
    CHECK (LENGTH(password_hash) >= 60); -- For bcrypt hashes

-- Prevent SQL injection in stored procedures
CREATE OR REPLACE FUNCTION safe_user_search(search_term TEXT) 
RETURNS TABLE (user_id INTEGER, username VARCHAR) AS $$
BEGIN
    RETURN QUERY
    EXECUTE 'SELECT user_id, username FROM secure_users WHERE username ILIKE $1'
    USING search_term;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
