using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;

namespace CUDAFingerprinting.Common.OrientationField.Tests
{
	[TestClass]
	public class FieldOrientationMatrixTest
	{
        [TestMethod]
	    public void GetOrientationMatrixTest()
	    {
	        var bytes = ImageHelper.LoadImageAsInt(Resources.SampleFinger);

	        OrientationField field = new OrientationField(bytes);
	        var res = field.GetOrientationMatrix(bytes.GetLength(0), bytes.GetLength(1));
	        FileInfo f = new FileInfo("Mytext.txt");
            StreamWriter w = f.CreateText();
            int a = res.GetLength(0);
            int b = res.GetLength(1);
	        for (int i = 0; i < res.GetLength(0); i++)
	        {
                for (int j = 0; j < res.GetLength(1); j++)
                {
                    w.Write(res[i,j]);
                    w.Write(" ");
                }
                w.WriteLine();
	        }
	        //field.SaveAboveToFile(image, Path.GetTempPath() + Guid.NewGuid() + ".bmp", true);
            w.Close();
	    }
    }
}