<meta charset="utf-8">
<html>
<head>
	<title>그림일기 분석 프로젝트</title>
	<meta name="description" content="Tis is the description">
	<meta charset="utf-8">
	<link rel="stylesheet" href="styles.css">
</head>
<?php
		exec("C:/Bitnami/wampstack-5.4.40-0/apache2/htdocs/total_program/venv/Scripts/python drawingAnalyze.py", $output);
    print_r(error_get_last());
?>

<meta charset="utf-8">
<html>
<head>
    <title>그림일기 분석 프로젝트</title>
    <meta charset="utf-8">
    <style>
        .header {
            font-weight: bold;
            text-align: center;
        }

        .cols {
            display: inline-block;
            width: 20%;
						align-self: center;
        }

        .cols img {
						align-content: center;
            width: 80%;
            margin: 10%;
        }
        .body {
            margin: 0 20px;
        }

				.image-text {
		        text-align: center;
						margin: 5px;
							 }
 				.image-text2 {
						text-align: center;
						font-size:1.25em;
						font-weight: bold;
						padding-bottom: 20px;
						margin-top: 0px;
					}

        .result {
            margin: 30px 0;
            text-align: center;

        }
    </style>
</head>
<body>
    <div class="header">
        <h1>[결과]</h1>
        <h2>부정적 그림일 확률</h2>
    </div>
    <div class="body">
<?php
    $img = 0;

    if ($handle = opendir("uploads")) {
        while (false !== ($entry = readdir($handle))) {
            if ($entry != '.' && $entry != '..'){
                if(strchr($entry, ".jpeg") == true) {
                    echo "<div class='cols'>
                            <img src='./uploads/$entry' width='150' height='250'/>
														<p class='image-text'>$entry <br/></p>
                            <p class='image-text2'>{$output[$img]}</p>
                        </div>";
                    $img++;
                }
            }
        }
        closedir($handle);
    }

    echo '<div class="result"><h2>최종 결과: '.$output[count($output)-1].'</h2></div>';

			// sleep(5);
			// $folder_path = "uploads"; //폴더 비우기
			// $files = glob($folder_path.'/*');
			// foreach($files as $file) {
			// 		if(is_file($file))
			// 		unlink($file);
			// 	}


?>
    </div>
</body>
</html>
