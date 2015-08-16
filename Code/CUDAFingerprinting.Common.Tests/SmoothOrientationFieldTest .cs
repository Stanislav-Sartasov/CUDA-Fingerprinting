using System;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class SmoothOrientationFieldTest
    {
        [TestMethod]
        public void SmoothOrientationTest()
        {
            var image = Resources.SampleFinger;
            var bytes = ImageHelper.LoadImage<int>(Resources.SampleFinger);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
            SmoothOrientationField SO_field = new SmoothOrientationField(field.Orientation);
            field.NewOrientation(SO_field.LocalOrientation());
            field.SaveAboveToFile(image);
        }
    }
}