﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.Collections.Generic;
using System.IO;

namespace CUDAFingerprinting.MinutiaDetection.Tests
{
    [TestClass]
    public class MinutiaDetectionTest
    {
        [TestMethod]
        public void MinutiaDetectorBasicTest()
        {
            var image = Resources.skeleton;
            var bytes = ImageHelper.LoadImageAsInt(image);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);

            List<Minutia> minutias = MinutiaDetector.GetMinutias(bytes, field);

            //field.SaveAboveToFile(image, Path.GetTempPath() + "//minutiaDetectionOrientationField.bmp", true);

            ImageHelper.MarkMinutiae(
                image,
                minutias,
                Path.GetTempPath() + "//minutiaDetectionMinutiae.png"
            );
            ImageHelper.MarkMinutiaeWithDirections(
                image,
                minutias,
                Path.GetTempPath() + "//minutiaDetectionMinutiaeWithDirections.png"
            );
        }
    }
}
