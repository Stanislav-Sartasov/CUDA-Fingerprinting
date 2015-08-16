using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.GPU.OrientationField.Test
{
	[TestClass]
	public class OrientationFieldTest
	{
		[DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "OrientationFieldInBlocks")]
		public static extern void OrientationFieldInBlocks(float[] orientation, float[] sourceBytes, int width, int height);
		[DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "OrientationFieldInPixels")]
		public static extern void OrientationFieldInPixels(float[] orientation, float[] sourceBytes, int width, int height);

		[TestMethod]
		public void OrientationAngleTest()
		{
            var image = Resources._1;
            var bytes = ImageHelper.LoadImage<int>(Resources._1);
			int height = bytes.GetLength(0);
            int width = bytes.GetLength(1);
			float[] sourceBytes = new float[height* width];
            float[] orientOut = new float[width * height];
            for (int i = 0; i < height - 1; i++)
			{
				for (int j = 0; j < width - 1; j++)
				{
					sourceBytes[i * width + j] = (float)bytes[i, j];
                    orientOut[i * width + j] = 0.0f;
				}
			}
            OrientationFieldInPixels(orientOut, sourceBytes, width, height);
            PixelwiseOrientationField field = new PixelwiseOrientationField(bytes, 16);
            double[,] orient_2D = new double[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    orient_2D[i, j] = orientOut[i * width + j];
            field.NewOrientation(orient_2D);
            field.SaveAboveToFile(image);
		}
	}
}
