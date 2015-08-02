using System;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.FeatureExtraction.SingularPoints;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class PoincareDetectionTest
    {
        [TestMethod]
        public void PoincareDetectionTestMethod()
        {
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources._44_8);
            int blockSize = 5;
            PoincareDetection.SingularityDetect(arrayI, blockSize).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
