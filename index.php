<html>
<head>
	<title>그림일기 분석 프로젝트</title>
	<meta name="description" content="Tis is the description">
	<meta charset="utf-8">
	<link rel="stylesheet" href="styles.css">
</head>
<body>
	<header class="main-header">
		<nav class="nav main-nav">
			<ul>
				<li><a href="index.php">HOME</a></li>
				<li><a href="test.php">TEST</a></li>
				<?php
					session_start();
				if(isset($_SESSION["userid"])) {
						//echo "<h2>{$_SESSION['userid']}님 환영합니다.</h2>";
						echo "<li><a href='logout.php'>LOGOUT</a></li>";
					} else {
						echo "<li><a href='login.html'>LOGIN</a></li>";
					}
				?>
			</ul>
		</nav>
		<h1 class="title">그림일기 분석 프로젝트</h1>
		<div class="container">
			<button class="btn btn-header" type="button" onclick="location.href='test.php'">분석하러 가기</button>
		</div>
	</header>
	<section class="content-section container">
			<h2 class="section-header">ABOUT</h2>
			<img class="about-band-image" src="Images/index_drawing.jpg">
			<p>한국에서 정신적 질병은 신체적 질병과 다르게 기피의 대상이 되며, 검사를 받기 쉽지 않습니다. 이는 감정 표현에 서툰 아이들의 경우에도 마찬가지입니다. 이 프로젝트에서는 딥러닝을 통해 아동이 그림일기에 표현한 부정적 감정 및 심리를 읽어내는 것을 목표로 합니다. 프로그램을 통해 부정적인 심리가 많이 나타난 아이는 심리상담 전문가에게 연결하여 치료를 받을 수 있도록 돕습니다.</p><br>
			<p>In Korea, we tend to cover up mental illness unlike physical illness and it's not easy to be examined. This is true for children who tends to not properly express their emotions. This project ains to read negative emotions and feelings expressed by children in their drawing diary through deep learning. Finally we will help children who need help by connecting them psychological counseling specialist to get treatment.</p>
			<br><br>
	</section>
	<footer class="main-footer">
		<div class="container main-footer-container">
			<h3 class="team-name">Team Cocacola</h3>
			<ul class="nav footer-nav">
				<li>
					<a href="mailto:yveltal@naver.com" target="_blank">
						<img src="Images/email_white.png">
					</a>
				</li>
			</ul>
		</div>
	</footer>
</body>
</html>
