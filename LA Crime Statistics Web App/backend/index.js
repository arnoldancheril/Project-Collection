const express = require("express");
const app = express();
const mysql = require("mysql");
const cloudSql = require("@google-cloud/sql");


var db = mysql.createConnection({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'Crime_Data',
    socketPath: process.env.DB_SOCKET || ''
})

// Database connection configured via environment variables:
// DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_SOCKET

app.get('/', (require, response) => {
    const sqlQuery = 'SELECT COUNT(*) AS weaponCount FROM Victims_table;';

    db.query(sqlQuery, (err, result) => {
        if (err) {
          console.error('Error executing query:', err);
          response.status(500).send('Internal Server Error 1');
          return;
        }
    
        // Assuming there is a single row with the count
        const weaponCount = result[0].weaponCount;
    
        // Send the count as a response
        response.send(`The count of rows in Weapons_table is: ${weaponCount}`);
      });
})

app.listen(3306, () => {
    console.log("running on port 3306");
})


// Might be an error in the above
// app.get('/', (req, res) => {
//     const sqlQuery = 'SELECT COUNT(*) AS weaponCount FROM Weapons_table;';

//     db.query(sqlQuery, (err, result) => {
//         if (err) {
//             console.error('Error executing query:', err);
//             res.status(500).send('Internal Server Error 1');
//             return;
//         }

//         const weaponCount = result[0].weaponCount;
//         res.send(`The count of rows in Weapons_table is: ${weaponCount}`);
//     });
// });



// app.get('/', (req, res) => {
//     // SQL query to get the count of rows from Weapons_table
//     const sqlQuery = 'SELECT COUNT(*) AS weaponCount FROM Weapons_table;';
  
//     db.query(sqlQuery, (err, result) => {
//       if (err) {
//         console.error('Error executing query:', err);
//         res.status(500).send('Internal Server Error 1');
//         return;
//       }
  
//       // Assuming there is a single row with the count
//       const weaponCount = result[0].weaponCount;
  
//       // Send the count as a response
//       res.send(`The count of rows in Weapons_table is: ${weaponCount}`);
//     });
//   });
  
//   const PORT = 3306; //process.env.PORT || 3002;
  
//   // Start the Express server
//   app.listen(PORT, () => {
//     console.log(`Server is running on port ${PORT}`);
//   });
  
//   // Close the database connection when the application is terminated
//   process.on('SIGINT', () => {
//     db.end((err) => {
//       if (err) {
//         console.error('Error closing the database connection:', err);
//       } else {
//         console.log('Database connection closed.');
//       }
//       process.exit();
//     });
//   });

// // app.listen(3002, () => {
// //     console.log("running on port 3002");
// // });
