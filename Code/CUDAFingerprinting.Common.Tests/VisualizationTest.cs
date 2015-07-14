using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;

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
            int[,] dummy = new int[OrientationField.DefaultSize * height, OrientationField.DefaultSize * width];
            var field = new OrientationField(dummy);

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
            field.SaveToFile(Path.GetTempPath() + Guid.NewGuid() + ".bmp", true);
        }


		[TestMethod]
		public void CompleteVisualizationTest()
		{
            var image = Resources.SampleFinger;
            var bytes = ImageHelper.LoadImageAsInt(Resources.SampleFinger);

			OrientationField field = new OrientationField(bytes);

            field.SaveAboveToFile(image, Path.GetTempPath() + Guid.NewGuid() + ".bmp", true);

            //for (int x = 0; x + 16 < image.Width; x += 16)
            //{
            //    Console.WriteLine("x = " + x);
            //    for (int y = 0; y + 16 < image.Height; y += 16)
            //    {
            //        Console.Write("y = " + y + " " + (int)(FingerPrint.GetOrientation(x, y) * 180 / Math.PI) + "\t");
            //    }
            //}

            // Visualization(image, FingerPrint);
		}
	}
}
