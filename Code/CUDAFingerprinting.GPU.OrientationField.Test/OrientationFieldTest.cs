using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.GPU.OrientationField.Test
{
	[TestClass]
	public class OrientationFieldTest
	{
		[DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "OrientationFieldInBlocks")]
		public static extern float[] OrientationFieldInBlocks(float[] floatArray, int width, int height);
		[DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "OrientationFieldInPixels")]
		public static extern float[] OrientationFieldInPixels(float[] floatArray, int width, int height);

		[TestMethod]
		public void OrientationAngleTest()
		{
			


			//char filepath[] = "C:\\temp\\1.bmp";
			//int width, height;
			//int* intBmpArray = loadBmp(filepath, &width, &height);
			//float* floatBmpArray = (float*)malloc(sizeof(float) * width * height);
			//for (int i = 0; i < width * height; i++){
			//	floatBmpArray[i] = (float)intBmpArray[i];
			//}
			//float* orientation;
			////orientation = OrientationFieldInBlocks(floatBmpArray, width, height);

			//orientation = OrientatiobFieldInPixels(floatBmpArray, width, height);

		


		}
	}
}
