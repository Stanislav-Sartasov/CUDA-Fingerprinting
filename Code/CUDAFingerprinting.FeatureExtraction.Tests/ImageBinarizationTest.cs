using System;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageProcessing.Binarization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class ImageBinarizationTest
    {
        [TestMethod]
        public void BinarializationTest()
        {
            double[,] arrayD = ImageHelper.LoadImage(Resources._1test);
            var binarizatedImageDouble = ImageBinarization.Binarize2D(arrayD, 128);
            for (int i = 0; i < arrayD.GetLength(0); i++)
            {
                for (int j = 0; j < arrayD.GetLength(1); j++)
                {
                    if (arrayD[i, j] < 128)
                    {
                        arrayD[i, j] = 0;
                    }
                    else
                    {
                        arrayD[i, j] = 255;
                    }

                    // Check binarizatedImageDouble
                    Assert.AreEqual(arrayD[i, j], binarizatedImageDouble[i, j]);
                }
            }
            ImageHelper.SaveArrayToBitmap(binarizatedImageDouble).Save(Path.GetTempPath()+ Guid.NewGuid() + ".bmp");
            
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources._2_6);
            var binarizatedImageInt = ImageBinarization.Binarize2D(arrayI, 128);
            for (int i = 0; i < arrayI.GetLength(0); i++)
            {
                for (int j = 0; j < arrayI.GetLength(1); j++)
                {
                    if (arrayI[i, j] < 128)
                    {
                        arrayI[i, j] = 0;
                    }
                    else
                    {
                        arrayI[i, j] = 255;
                    }

                    // Check binarizatedImageInt
                    Assert.AreEqual(arrayI[i, j], binarizatedImageInt[i, j]);
                }
            }
            ImageHelper.SaveArrayToBitmap(binarizatedImageInt).Save("d://Result.bmp");
        }
    }
}
