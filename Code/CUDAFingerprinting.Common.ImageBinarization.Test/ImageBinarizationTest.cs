using System;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common.ImageBinarization;
using System.IO;

namespace CUDAFingerprinting.Common.ImageBinarization.Test
{
    [TestClass]
    public class ImageBinarizationTest
    {
        [TestMethod]
        public void BinarializationTest()
        {
            double[,] arrayD = ImageHelper.LoadImage(Resources._2_6);
            var binarizatedImageDouble = ImageBinarization.Binarizator2D(arrayD, 128);
            ImageHelper.SaveArrayToBitmap(binarizatedImageDouble).Save(Path.GetTempPath()+ Guid.NewGuid() + ".bmp");
 
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources._2_6);
            var binarizatedImageInt = ImageBinarization.Binarizator2D(arrayI, 128);
            ImageHelper.SaveArrayToBitmap(binarizatedImageInt).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
