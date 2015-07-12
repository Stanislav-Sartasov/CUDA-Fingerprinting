using System;
using System.IO;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.AdaptiveBinarization.Test
{
    [TestClass]
    public class AdaptiveBinarizationTest
    {
        [TestMethod]
        public void AdaptiveBinarizationTestMethod()
        {
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources.test);
            
            OrientationField.OrientationField field = new OrientationField.OrientationField(arrayI);

            field.SaveAboveToFile(Resources.test, Path.GetTempPath() + Guid.NewGuid() + ".bmp", true);
            
            var binarizatedInt = AdaptiveBinarization.AdaptiveImageBinarization(arrayI);
            ImageHelper.SaveArrayToBitmap(binarizatedInt).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
