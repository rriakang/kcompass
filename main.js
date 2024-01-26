const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const session = require('express-session'); // express-session 추가
const db = require('./db');
const cors = require('cors');


const app = express();
const port = process.env.PORT || 8000;
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use('/public', express.static('public'));
app.use(express.urlencoded({ extended: true })); // Express 내장 body-parser 사용
app.use(cors());
app.use(express.json()); // json으로 받기 위해선 이 모듈을 정의를 해야함 !! 꼭 하기

// 세션 미들웨어 설정
app.use(session({
  secret: 'your-secret-key', // 세션을 암호화하기 위한 시크릿 키
  resave: false,
  saveUninitialized: true
}));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, './views/main.html'));
});

app.get('/main', (req, res) => {
  res.sendFile(path.join(__dirname, './views/gotest.html'));
});

app.get('/administor', (req, res) => {
  res.sendFile(path.join(__dirname, './views/administor.html'));
});

app.post('/administor', (req, res) => {
  const body = req.body;
  const id = body.id;
  const pw = body.pas;

  // 데이터베이스에서 해당 ID의 사용자 정보를 가져옴
  // 엔터누르면 로그인 회원가입 안됌 이부분 개선 필요
  db.query('SELECT * FROM admin WHERE id = ?', [id], (err, data) => {
    if (err) {
      console.error(err);
      res.status(500).send('Internal Server Error');
      return;
    }

    console.log('입력한 ID:', id);
    console.log('데이터베이스에서 가져온 데이터:', data);

    if (data.length === 0 || pw !== data[0].password) {
      console.log('로그인 실패');
      res.redirect('/administor');
    } else {
      console.log('로그인 성공');

      // 세션에 사용자 정보 추가
      req.session.is_logined = true;
      req.session.name = data[0].name;
      req.session.id = data[0].id;

      res.redirect('/data'); // '/data'로 리다이렉션
    }
  });
});


// 로그인 후 (관리자 페이지)

// app.get('/data', (req, res) => {
//   // 세션에 사용자 정보가 있는지 확인
//   if (req.session && req.session.is_logined) {
//     res.sendFile(path.join(__dirname, './views/data.html'));
//   } else {
//     // 세션이 없다면 로그인 페이지로 리다이렉트
//     res.redirect('/administor');dlr
//   }
// });

app.get('/data', (req, res) => {
  // 세션에 사용자 정보가 있는지 확인
  if (req.session && req.session.is_logined) {
    // 페이지와 관련된 변수 설정
    const itemsPerPage = 50; // 한 페이지에 표시할 항목 수
    const currentPage = req.query.page || 1; // 현재 페이지 번호 (기본값: 1)

    // MySQL 테이블에서 전체 데이터 개수 가져오기
    db.query('SELECT COUNT(*) as count FROM test_1', (countErr, countResult) => {
      if (countErr) {
        console.error(countErr);
        res.status(500).send('Internal Server Error');
        return;
      }

      const totalCount = countResult[0].count;
      const totalPages = Math.ceil(totalCount / itemsPerPage); // 전체 페이지 수

      // 현재 페이지에 해당하는 데이터 가져오기
      const offset = (currentPage - 1) * itemsPerPage;
      const limit = itemsPerPage;

      db.query('SELECT * FROM test_1 LIMIT ?, ?', [offset, limit], (dataErr, data) => {
        if (dataErr) {
          console.error(dataErr);
          res.status(500).send('Internal Server Error');
          return;
        }

        // 데이터를 data.html에 전달하여 렌더링
        res.render('data', { test1Data: data, currentPage, totalPages, itemsPerPage });
      });
    });
  } else {
    // 세션이 없다면 로그인 페이지로 리다이렉트
    res.redirect('/administor');
  }
});


app.get('/block1', (req, res) => {
  res.sendFile(path.join(__dirname, './views/block1.html'));
});

app.post('/block1', (req, res) => {
  const formData = {
      Age: parseInt(req.body.Age), // Age를 정수로 변환
      Sex_MF: parseInt(req.body.Sex_MF),
      Marriage: parseInt(req.body.Marriage),
      Employment: parseInt(req.body.Employment),
      Religion: parseInt(req.body.Religion),
      Cohabitant: parseInt(req.body.Cohabitant),
      UrbanRural: parseInt(req.body.UrbanRural),
      NP_Family_Hx: parseInt(req.body.NP_Family_Hx),
      Med_Hx: parseInt(req.body.Med_Hx),
      NP_Hx: parseInt(req.body.NP_Hx),
      // 나머지 필드에 대한 수집 추가
  };

  const columns = Object.keys(formData).join(', ');
  const values = Object.values(formData).map(value => db.escape(value)).join(', ');
  const sql = `INSERT INTO test_1 (${columns}) VALUES (${values})`;

  db.query(sql, (err, results) => {
      if (err) {
          console.error('MySQL query error:', err); // 에러 메시지 출력
          res.status(500).json({ error: 'Internal Server Error' });
          return;
      }

      res.status(200).json({ message: 'Data saved successfully' });
  });
  //res.redirect('/predict');
});

//내일 할일 모델 예측 redirect하기

app.get('/predict',(req,res) => {
  res.sendFile(path.join(__dirname, './views/predict.html'))
});


// const model = require('./machin-learning-model') //머신러닝 
// app.get('/predict',(req,res) => {
//   const data = req.query.data;
//   const prediction = model(data);
//   res.json({prediction});
// });




app.get('/block2', (req, res) => {
  res.sendFile(path.join(__dirname, './views/block2.html'));
});

app.get('/block3', (req, res) => {
  res.sendFile(path.join(__dirname, './views/block3.html'));
});

app.listen(port, () => {
  console.log(`서버가 http://localhost:${port}/ 에서 실행 중입니다.`);
});

