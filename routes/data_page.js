// routes/data.js
const express = require('express');
const router = express.Router();

// 해당 라우트에서 사용할 데이터베이스 연결 객체 (예제에서는 db라 가정)
const db = require('./db');

router.get('/', (req, res) => {
  // 페이지 번호 가져오기
  const page = parseInt(req.query.page) || 1;
  const itemsPerPage = 10; // 페이지당 아이템 수

  // 데이터베이스에서 페이지에 해당하는 데이터 조회
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;

  db.query('SELECT * FROM test_1', (err, data) => {
    if (err) {
      console.error(err);
      res.status(500).send('Internal Server Error');
      return;
    }

    const totalPages = Math.ceil(data.length / itemsPerPage);
    const currentPageData = data.slice(startIndex, endIndex);

    res.render('data', { test1Data: currentPageData, currentPage: page, totalPages: totalPages });
  });
});

module.exports = router;
