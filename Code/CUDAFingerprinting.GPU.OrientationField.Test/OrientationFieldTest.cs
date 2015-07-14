using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;



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

			var image = Properties.Resources._1;
			var bytes = ImageHelper.LoadImageAsInt(Properties.Resources._1);
			int height = bytes.GetUpperBound(0) + 1;
			int width = bytes.GetUpperBound(1) + 1;
			float[] sourceBytes = new float[height* width];
			float[] field = new float[height * width];
			for (int i = 0; i < height - 1; i++)
			{
				for (int j = 0; j < width - 1; j++)
				{
					sourceBytes[i * width + j] = (float)bytes[i, j];
					field[i * width + j] = 0.0f;
				}

			}

			OrientationFieldInBlocks(field, sourceBytes, width, height);


		}
	}
}
