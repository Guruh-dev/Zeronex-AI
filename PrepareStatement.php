<?php
namespace Numpy\Database;

use \PDO;
use \PDOException;

class PrepareStatement {
    private $pdo;
    private $statement;
    private $error;
    private $lastInsertId;

    public function __construct($pdo) {
        $this->pdo = $pdo;
        $this->error = null;
        $this->lastInsertId = null;
    }

    /**
     * Prepare a SQL statement
     * @param string $sql
     * @return PrepareStatement
     * @throws 'PDOException'
     */
    public function prepare($sql) {
        try {
            $this->statement = $this->pdo->prepare($sql);
            return $this;
        } catch (PDOException $e) {
            $this->error = $e->getMessage();
            throw $e;
        }
    }

    /**
     * Bind parameters to the prepared statement
     * @param array $params
     * @return PrepareStatement
     * @throws 'PDOException'
     */
    public function bindParams($params) {
        try {
            foreach ($params as $key => $value) {
                $this->statement->bindValue(
                    is_numeric($key) ? $key + 1 : $key,
                    $value,
                    $this->getDataType($value)
                );
            }
            return $this;
        } catch (PDOException $e) {
            $this->error = $e->getMessage();
            throw $e;
        }
    }

    /**
     * Execute the prepared statement
     * @return bool
     * @throws PDOException
     */
    public function execute() {
        try {
            $result = $this->statement->execute();
            $this->lastInsertId = $this->pdo->lastInsertId();
            return $result;
        } catch (PDOException $e) {
            $this->error = $e->getMessage();
            throw $e;
        }
    }

    /**
     * Fetch all results
     * @param int $fetchMode Optional fetch mode
     * @return array
     */
    public function fetchAll($fetchMode = PDO::FETCH_ASSOC) {
        return $this->statement->fetchAll($fetchMode);
    }

    /**
     * Fetch single row
     * @param int $fetchMode Optional fetch mode
     * @return array|false
     */
    public function fetch($fetchMode = PDO::FETCH_ASSOC) {
        return $this->statement->fetch($fetchMode);
    }

    /**
     * Count affected rows
     * @return int
     */
    public function rowCount() {
        return $this->statement->rowCount();
    }

    /**
     * Get last insert ID
     * @return string|false
     */
    public function getLastInsertId() {
        return $this->lastInsertId;
    }

    /**
     * Get the last error message
     * @return string|null
     */
    public function getError() {
        return $this->error;
    }

    /**
     * Get the data type for binding parameters
     * @param mixed $value
     * @return int
     */
    private function getDataType($value) {
        switch (true) {
            case is_int($value):
                return PDO::PARAM_INT;
            case is_bool($value):
                return PDO::PARAM_BOOL;
            case $value === null:
                return PDO::PARAM_NULL;
            case is_resource($value):
                return PDO::PARAM_LOB;
            default:
                return PDO::PARAM_STR;
        }
    }

    /**
     * Close the cursor
     * @return bool
     */
    public function closeCursor() {
        return $this->statement->closeCursor();
    }
}

