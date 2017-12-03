 /** Implementacao modificada do classificador em cascata.
 As alteracoes com relacao ao presente nas referencias do opencv
 sao comentadas. */ 
 
 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include <opencv2/opencv.hpp>
 #include <math.h>       /** biblioteca para uso da funcao arctan */

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 #define PI 3.14159265 /** define o valor de pi para conversao de radiano para grau */

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Roll determination"; /** nome da janela */
 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream
   capture = cvCaptureFromCAM( -1 );
   if( capture )
   {
   while( true )
   {
   frame = cvQueryFrame( capture );
   

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;
  char angle_str[5]; /** variavel para receber o valor do angulo em uma string */
  double param; /** armazena o parametro de entrada da funcao arctan */
  double x; /** coordenada x do olho */
  double y; /** coordenada y do olho */
  double posicoes[4]; /** armazena os dois pares de coordenadas dos olhos */

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    int n = 0; /** variavel para contar qual o olho com a coordenada armazenada */

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( x = faces[i].x + eyes[j].x + eyes[j].width*0.5, y = faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
       posicoes[n] = x; /** armazena a coordenada x do olho */
       posicoes[n+2] = y; /** armazena a coordenada y do olho */
       n = n+1; /** incrementa o valor de n para registrar o proximo olho */
     }
  }
  //-- Show what you got
  
  cv::Size s = frame.size();
  int rows = s.height; /** conta as linhas */
  int cols = s.width; /** conta as colunas */
  param = (posicoes[2]-posicoes[3])/(posicoes[0]-posicoes[1]); /** argumento da funcao arctan */
  double angle = atan (param) * 180 / PI; /** obtem o angulo */
  sprintf(angle_str, "%.4f", angle); /** converte o valor double do angulo para string */
  /** cria um retangulo de fundo para o texto com o angulo */
  rectangle(frame, Point(floor(cols/2.5), floor(0.8*rows)), Point(floor(cols-(cols/2.5)), floor(0.9*rows)), Scalar( 0, 0, 0 ), -1, 8 );
  /** escrever no frame o valor do angulo */
  putText(frame, angle_str, Point(floor(cols/2.5), floor(0.85*rows)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1.0);
  imshow( window_name, frame ); /** exibe a imagem da camera com os circulos o retangulo e o texto */
 }
