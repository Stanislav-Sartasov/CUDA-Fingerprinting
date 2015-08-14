using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.FeatureExtraction;
using CUDAFingerprinting.ImageProcessing;
using CUDAFingerprinting.TemplateMatching;

namespace TriangulationWithAfineTransformation
{
    public partial class MainForm : Form
    {
        private static Pen defPointPen = new Pen(Color.Red, 2);
        private static Pen defLinePen = new Pen(Color.LimeGreen, 2);

        private static Graphics graphic;
        private static int formHeight;

        public MainForm()
        {
            InitializeComponent();

            graphic = this.CreateGraphics();
            formHeight = this.Size.Height;
        }

        private void OnMainFormResize(object sender, EventArgs e)
        {
            formHeight = this.Size.Height;
            graphic = this.CreateGraphics();
            graphic.Clear(this.BackColor);
        }

        
        private void LoadImage(PictureBox pictureBox) 
        {
            using (OpenFileDialog dlg = new OpenFileDialog())
            {
                dlg.Title = "Open Image";
                dlg.Filter = "bmp files (*.bmp)|*.bmp";

                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    // Create a new Bitmap object from the picture file on disk,
                    // and assign that to the PictureBox.Image property
                    pictureBox.Image = new Bitmap(dlg.FileName);
                }
            }
        }

        private void ImageToOnClick(object sender, EventArgs e)
        {
            LoadImage(ImageTo);
        }

        private void ImageFromOnClick(object sender, EventArgs e)
        {
            LoadImage(ImageFrom);
        }

        private List<Minutia> GetMinutiasFrom(PictureBox box){
            var image = box.Image;

            Bitmap bmp = (Bitmap) box.Image;

            var bytes = ImageHelper.LoadImage<int>((Bitmap)box.Image);
            CUDAFingerprinting.Common.OrientationField.PixelwiseOrientationField field = new CUDAFingerprinting.Common.OrientationField.PixelwiseOrientationField(bytes, 16);

            return CUDAFingerprinting.FeatureExtraction.Minutiae.MinutiaDetector.GetMinutias(bytes, field);
        }

        public Bitmap Binarize(Bitmap src, int threshold)
        {
            Bitmap bmp = new Bitmap(src.Width, src.Height);
            for (int i = 0; i < bmp.Width; i++)
            {
                for (int j = 0; j < bmp.Height; j++)
                {
                    bmp.SetPixel(i, j, src.GetPixel(i, j).B < threshold ? Color.Black : Color.White);
                }
            }
            return bmp;
        }

        private void MatchButtonClick(object sender, EventArgs e)
        {
           

            List<Minutia> minutiasFrom = GetMinutiasFrom(ImageFrom);
            
            MessageBox.Show("Found " + minutiasFrom.Count + " minutias in left picture.");

            List<Classes.Point> pointsFrom = new List<Classes.Point>();
            Graphics ImageFromGraphics = ImageFrom.CreateGraphics();
            foreach (Minutia m in minutiasFrom)
            {
                Classes.Point p = new Classes.Point(m.X, ImageFrom.Height - m.Y+1);
                p.Paint(ImageFromGraphics, defPointPen, ImageFrom.Height);
                pointsFrom.Add(p);
            }

            if (minutiasFrom.Count > 500)
            {
                string messageBoxText = "Do you really want to continue?\n You'll just waste your time...";
                string caption = "Too many minutuias";
                MessageBoxButtons mb = MessageBoxButtons.OKCancel;
                if (MessageBox.Show(messageBoxText, caption, mb) == System.Windows.Forms.DialogResult.Cancel)
                    return;
            }

            Classes.TriangulationBuilder tb = new Classes.TriangulationBuilder(pointsFrom);
            MessageBox.Show("Triangulation built! It has: "+tb.triangles.Count+" triangles.");

            List<Classes.Triangle> triangulationFrom = tb.triangles;
            foreach (Classes.Triangle t in tb.triangles) {
                t.Paint(ImageFromGraphics, defLinePen, defPointPen, ImageFrom.Height);
            }
            
            List<Minutia> minutiasTo = GetMinutiasFrom(ImageTo);

            MessageBox.Show("Found " + minutiasTo.Count + " minutias in right picture");

            List<Classes.Point> pointsTo = new List<Classes.Point>();


            Graphics ImageToGraphics = ImageTo.CreateGraphics();
            foreach (Minutia m in minutiasTo)
            {
                Classes.Point p = new Classes.Point(m.X, ImageTo.Height - m.Y+1);
                p.Paint(ImageToGraphics, defPointPen, ImageTo.Height);
                pointsTo.Add(p);
            }

            if (minutiasTo.Count > 500)
            {
                string messageBoxText = "Do you really want to continue?\n You'll just waste your time...";
                string caption = "Too many minutuias";
                MessageBoxButtons mb = MessageBoxButtons.OKCancel;
                if (MessageBox.Show(messageBoxText, caption, mb) == System.Windows.Forms.DialogResult.Cancel)
                    return;
            }

            Classes.TriangulationBuilder tbTo;
            if (minutiasTo.Count < 500)
                tbTo = new Classes.TriangulationBuilder(pointsTo);
            else
                tbTo = new Classes.TriangulationBuilder(pointsTo, ImageToGraphics, defLinePen, defPointPen, new Pen(Color.Gold, 1), ImageTo.Height, 0);
            
            MessageBox.Show("Triangulation built! It has: " + tbTo.triangles.Count + " triangles.");

            List<Classes.Triangle> triangulationTo = tbTo.triangles;
            foreach (Classes.Triangle t in tbTo.triangles)
            {
                t.Paint(ImageToGraphics, defLinePen, defPointPen, ImageFrom.Height);
            }

            Classes.TriangulationsMatcher tm = new Classes.TriangulationsMatcher(triangulationFrom, triangulationTo);
            double[] result = tm.Match(0.001,90);
            MessageBox.Show("Equals: " + result[0] + ", very close:" + result[1]);
        }
    }
}
