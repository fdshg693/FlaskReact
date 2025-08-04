// API functions for SQL operations

/**
 * Execute SQL query against the database
 * @param {string} query - SQL query to execute
 * @param {string} database - Database identifier
 * @returns {Promise<Object>} Query execution results
 */
async function executeSqlQuery(query, database = 'sample') {
  try {
    const response = await fetch('/api/sql/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: query,
        database: database
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error executing SQL query:', error);
    throw error;
  }
}

/**
 * Get AI assistance for SQL query
 * @param {string} question - Natural language question about SQL
 * @param {string} schema - Database schema information
 * @returns {Promise<Object>} AI assistance response
 */
async function getAiSqlAssistance(question, schema = '') {
  try {
    const response = await fetch('/api/sql/ai-assist', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        question: question,
        schema: schema
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting AI SQL assistance:', error);
    throw error;
  }
}

/**
 * Get database schema information
 * @param {string} database - Database identifier
 * @returns {Promise<Object>} Database schema
 */
async function getDatabaseSchema(database = 'sample') {
  try {
    const response = await fetch(`/api/sql/schema/${database}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting database schema:', error);
    throw error;
  }
}

/**
 * Get SQL practice questions
 * @param {string} difficulty - Difficulty level (beginner, intermediate, advanced)
 * @returns {Promise<Object>} Practice questions
 */
async function getSqlQuestions(difficulty = 'beginner') {
  try {
    const response = await fetch(`/api/sql/questions/${difficulty}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting SQL questions:', error);
    throw error;
  }
}

/**
 * Validate SQL query syntax
 * @param {string} query - SQL query to validate
 * @returns {Promise<Object>} Validation results
 */
async function validateSqlQuery(query) {
  try {
    const response = await fetch('/api/sql/validate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: query
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error validating SQL query:', error);
    throw error;
  }
}