var mysql = require('mysql');
var db = mysql.createConnection({
    host: '127.0.0.1',
    user: 'root',
    port: 3305,  // MySQL의 기본 포트는 3306입니다.
    password: 'Xptmxm1212!@',
    database: 'admin'  // 오타가 있던 'adminisrator'를 'administrator'로 수정했습니다.
});
db.connect();

module.exports = db;
