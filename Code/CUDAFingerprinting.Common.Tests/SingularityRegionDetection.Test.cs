//using System;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using CUDAFingerprinting.Common;
//using System.Drawing;

//namespace CUDAFingerprinting.Common.SingularityRegionDetection.Test
//{
//    [TestClass]
//    public class SingularityRegionDetectionTests
//    {
//        [TestMethod]
//        public void SingularityRegionDetectionTest()
//        {
//            var image = Resources._8_2;
//            var intBmp = ImageHelper.LoadImageAsInt(Resources._8_2);

//            PixelwiseOrientationField field = new PixelwiseOrientationField(intBmp, 8);

//            field.SaveAboveToFile(image, "Orientation.bmp", true);

//            int width = intBmp.GetLength(0);
//            int height = intBmp.GetLength(1);
            
//            var dAr = field.Orientation;

//            System.Console.WriteLine(dAr.GetLength(0));
//            System.Console.WriteLine(dAr.GetLength(1));

//            SingularityRegionDetection D = new SingularityRegionDetection(dAr);

//            double[,] Result = D.Detect(dAr);
//            double[,] revertedResult = new double[height, width];
            
//            for (int i = 0; i < width; i++)
//            {
//                for (int j = 0; j < height; j++)
//                {
//                    revertedResult[height - 1 -j, i] = Result[i, j];
//                }
//            }

//            Bitmap bmp = D.MakeBitmap(revertedResult);
//            bmp.Save("Result.jpg");
//        }
//    }
//}
