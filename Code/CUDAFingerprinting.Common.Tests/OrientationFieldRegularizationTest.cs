using System;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
  [TestClass]
  public class OrientationFieldRegularizationTest
  {
    [TestMethod]
    public void OrientationRegularizationTest()
    {
      var image = Resources.SampleFinger;
      var bytes = ImageHelper.LoadImage<int>(Resources.SampleFinger);
      PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
      OrientationFieldRegularization new_field = new OrientationFieldRegularization(field.Orientation, 25);
      field.NewOrientation(new_field.LocalOrientation());
      field.SaveAboveToFile(image);
    }
  }
}
