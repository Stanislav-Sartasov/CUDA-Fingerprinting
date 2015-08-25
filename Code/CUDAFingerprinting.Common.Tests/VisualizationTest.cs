using System;
using CUDAFingerprinting.Common.OrientationField;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
	[TestClass]
	public class VisualizationTest
	{
        [TestMethod]
        public void SimpleVisualizationTest()
        {
            var height = 7;
            var width = 6;
            int[,] dummy = new int[OrientationField.OrientationField.DefaultSize * height, OrientationField.OrientationField.DefaultSize * width];
            var field = new OrientationField.OrientationField(dummy);

            for(int column=0;column<width;column++)
            {
                field.Blocks[0, column].Orientation = -Math.PI / 2;
                field.Blocks[1, column].Orientation = -Math.PI / 3;
                field.Blocks[2, column].Orientation = -Math.PI / 6;
                field.Blocks[3, column].Orientation = 0;
                field.Blocks[4, column].Orientation = Math.PI / 6;
                field.Blocks[5, column].Orientation = Math.PI / 3;
                field.Blocks[6, column].Orientation = Math.PI / 2;
            }
            field.SaveToFile();
        }


		[TestMethod]
		public void CompleteVisualizationTest()
		{
            var image = Resources.SampleFinger;
            var bytes = ImageHelper.LoadImage<int>(Resources.SampleFinger);

			var field = new OrientationField.OrientationField(bytes);

            field.SaveAboveToFile(image);
		}
	}
}
