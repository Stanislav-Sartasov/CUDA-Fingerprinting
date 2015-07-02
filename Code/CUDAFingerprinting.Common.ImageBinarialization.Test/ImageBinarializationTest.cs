using System;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.ImageBinarialization;
using System.IO;

namespace CUDAFingerprinting.Common.ImageBinarialization.Test
{
    [TestClass]
    public class ImageBinarializationTest
    {
        [TestMethod]
        public void BinarializationTest()
        {
            Bitmap img = Resources._2_6;
            var resultImage = new ImageBinarialization(img);
            resultImage.Binarizator(resultImage, 128).Save("ResultImage.bmp", ImageFormat.Bmp);
        }
    }
}
