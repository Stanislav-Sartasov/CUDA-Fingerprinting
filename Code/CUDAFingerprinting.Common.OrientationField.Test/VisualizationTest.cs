using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;

namespace CUDAFingerprinting.Common.OrientationField.Tests
{
	[TestClass]
	public class VisualizationTest
	{

		[TestMethod]
		public void TestVisualization()
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
