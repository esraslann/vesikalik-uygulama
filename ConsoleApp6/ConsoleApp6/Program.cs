using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;
using OpenCvSharp;

class Program
{
    static void Main(string[] args)
    {
        // Fotoğrafı yükle
        string imagePath = "C:\\Users\\esra\\Desktop\\vesikalık-A.-Karaman.jpg"; // Dosya yolunu doğrudan belirtin
        Image<Bgr, Byte> inputImage = new Image<Bgr, byte>(imagePath);

        // Yüz tanıma için hazırlık
        var faceDetector = new Emgu.CV.CascadeClassifier("haarcascade_frontalface_default.xml");

        // Giriş fotoğrafında yüz tespiti
        var faces = faceDetector.DetectMultiScale(inputImage, 1.1, 10, System.Drawing.Size.Empty);

        if (faces.Length > 0)
        {
            // İlk bulunan yüzü alın
            var face = faces[0];

            // Yüz bölgesini kırp
            var faceRegion = inputImage.GetSubRect(face);

            // Vesikalık boyutları
            System.Drawing.Size vesikalikSize = new System.Drawing.Size(413, 531);

            // Yüzün çerçeveye oranı
            double faceArea = face.Width * face.Height;
            double imageArea = inputImage.Width * inputImage.Height;
            double faceToFrameRatio = faceArea / imageArea;

            // Arka plan sadeliği (ortalama renk varyasyonu)
            var grayImage = inputImage.Convert<Gray, byte>();
            var blurredImage = grayImage.SmoothGaussian(21);
            var diffImage = new Image<Gray, byte>(grayImage.Size);
            CvInvoke.AbsDiff(grayImage, blurredImage, diffImage);
            double backgroundSimplicity = CvInvoke.Mean(diffImage).V0;

            // Benzerlik skorunu hesapla
            double similarityScore = CalculateSimilarity(faceToFrameRatio, backgroundSimplicity);

            Console.WriteLine($"Fotoğraf vesikalık olma benzerlik skoru: {similarityScore:F2}%");
        }
        else
        {
            Console.WriteLine("Fotoğrafta yüz tespit edilemedi.");
        }
        Console.ReadLine();
    }

    static double CalculateSimilarity(double faceToFrameRatio, double backgroundSimplicity)
    {
        // Basit bir model: Yüzün çerçeveye oranı %30-70 aralığında olmalı
        double faceRatioScore = 0;
        if (faceToFrameRatio >= 0.3 && faceToFrameRatio <= 0.7)
        {
            faceRatioScore = 100;
        }
        else if (faceToFrameRatio < 0.3)
        {
            faceRatioScore = faceToFrameRatio / 0.3 * 100;
        }
        else
        {
            faceRatioScore = (1 - (faceToFrameRatio - 0.7) / 0.3) * 100;
        }

        // Arka plan sadeliği düşük olmalı (rastgele bir eşik değeri kullanıyoruz, örn: 50)
        double simplicityScore = Math.Max(0, 100 - backgroundSimplicity);

        // Genel benzerlik skoru (yüz oranı ve sadelik skorlarının ortalaması)
        return (faceRatioScore + simplicityScore) / 2;
    }
}