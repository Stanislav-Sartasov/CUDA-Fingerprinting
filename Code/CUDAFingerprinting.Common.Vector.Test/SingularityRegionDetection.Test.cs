using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.Drawing;

namespace CUDAFingerprinting.Common.SingularityRegionDetection.Test
{
    [TestClass]
    public class SingularityRegionDetectionTests
    {
        [TestMethod]
        public void SingularityRegionDetectionTest()
        {
            var image = Properties.Resources.SampleFinger;
            var intBmp = ImageHelper.LoadImageAsInt(Properties.Resources.SampleFinger);

            PixelwiseOrientationField pxl = new PixelwiseOrientationField(intBmp, 16);

            //pxl.SaveAboveToFile(image, "PixelWiseOnSegmented.bmp", true);

            int width = intBmp.GetLength(0);
            int height = intBmp.GetLength(1);
            
            double[,] dAr = new double[width, height];

            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                {
                    dAr[i, j] = (pxl.GetOrientation(i, j) < 0 && pxl.GetOrientation(i, j) >= -Math.PI / 2) ? 
                        pxl.GetOrientation(i, j) + (Math.PI) : pxl.GetOrientation(i, j);
                }

            SingularityRegionDetection D = new SingularityRegionDetection(dAr);

            double[,] Result = D.Detect(dAr);
            double[,] revertResult = new double[height, width];

            Bitmap bmp = D.MakeBitmap(Result);
            bmp.Save("Result.jpg");
        }
    }
}
