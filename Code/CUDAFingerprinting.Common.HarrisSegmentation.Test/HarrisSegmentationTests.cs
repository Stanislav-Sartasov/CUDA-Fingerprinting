using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.Common.HarrisSegmentation.Test
{
    [TestClass]
    public class HarrisSegmentationTests
    {
        [TestMethod]
        public void HarrisSegmentationTest()
        {
            var image = Properties.Resources._52_8;
            HarrisSegmentation M = new HarrisSegmentation(image);

            double[,] matrix = M.GaussFilter();
            byte[,] byteMatrix = M.Segmentate(matrix);

            string filename = "Result.jpg";

            var bmp = M.MakeBitmap(byteMatrix);
            M.SaveSegmentation(bmp, filename);

            image.Dispose();
            bmp.Dispose();
        }
    }
}
