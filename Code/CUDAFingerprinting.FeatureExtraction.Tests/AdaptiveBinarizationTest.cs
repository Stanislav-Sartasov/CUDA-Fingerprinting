﻿using System;
using System.IO;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.ImageProcessing.Binarization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class AdaptiveBinarizationTest
    {
        [TestMethod]
        public void AdaptiveBinarizationTestMethod()
        {
            int[,] arrayI = ImageHelper.LoadImage<int>(Resources._1test);
            var arr = AdaptiveBinarization.AdaptiveBinarize(arrayI);
            ImageHelper.SaveArrayAndOpen(arr);
        }
    }
}
