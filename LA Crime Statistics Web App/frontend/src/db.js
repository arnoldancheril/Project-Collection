const mysql = require('mysql2');

const connection = mysql.createConnection({
  host: 'your-hostname',
  port: 3306,
  user: 'your-username',
  password: 'your-password',
  database: 'your-database'
});

module.exports = connection;