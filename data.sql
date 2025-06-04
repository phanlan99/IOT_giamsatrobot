CREATE TABLE users (
    id SERIAL PRIMARY KEY,                    
    username VARCHAR(50) NOT NULL UNIQUE,    
    password VARCHAR(255) NOT NULL
);


------thêm------
INSERT INTO users (username, password)
VALUES ('alice', 'hashed_password_123');

--- phân loại ---

CREATE TABLE supplement_data (
    id SERIAL PRIMARY KEY,
    record_time TIMESTAMP NOT NULL,
    beroca INTEGER,
    cachua INTEGER,
    cam INTEGER,
    egg INTEGER,
    maleutyl INTEGER,
    probio INTEGER,
    sui INTEGER,
    topralsin INTEGER,
    vitatrum INTEGER,
    zidocinDHG INTEGER
);



