using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace CUDAFingerprinting.Common.Tests
{
	[TestClass]
	public class PixelwiseOrientationFieldTest
	{
		[TestMethod]
		public void PixelwiseOrientationTest()
		{
			var image = Resources.SampleFinger;
			var bytes = ImageHelper.LoadImage<int>(Resources.SampleFinger);

			PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
			
			//double orientation = field.GetOrientation(1, 1);
			field.SaveAboveToFile(image);
		}
	}
}
