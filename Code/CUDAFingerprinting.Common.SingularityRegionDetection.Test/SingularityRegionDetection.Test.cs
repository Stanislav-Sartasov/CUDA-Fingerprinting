using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.Common.SingularityRegionDetection.Test
{
    [TestClass]
    public class SingularityRegionDetectionTests
    {
        [TestMethod]
        public void SingularityRegionDetectionTest()
        {
            var image = Properties.Resources._8_2;
            var intBmp = ImageHelper.LoadImageAsInt("..//8_2.bmp");
            PixelwiseOrientationField pxl  = new PixelwiseOrientationField(intBmp, 32);

            double[,] dAr = new double[intBmp.GetLength(0), intBmp.GetLength(1)];
            for (int i = 0; i < intBmp.GetLength(0); ++i)
                for (int j = 0; j < intBmp.GetLength(1); ++j)
                    dAr[i, j] = pxl.GetOrientation(i, j);

            SingularityRegionDetection D = new SingularityRegionDetection(dAr);

            double[,,] Result = D.Detect(dAr);
        }
    }
}
