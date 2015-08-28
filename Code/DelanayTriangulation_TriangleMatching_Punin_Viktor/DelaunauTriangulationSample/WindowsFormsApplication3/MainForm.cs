using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using System.Linq;
using System.Diagnostics;
using DelaunauTriangulationSample.Classes;

namespace DelaunauTriangulation
{
    public partial class MainForm : Form
    {
        private static Pen defaultPointPen = new Pen(Color.Red, 2);
        private static Pen defaultLinePen = new Pen(Color.Green, 1);
        private static Pen defaultNewLintPen = new Pen(Color.Gold, 1);
        private List<DelaunauTriangulationSample.Classes.Point> points = null;
        private Graphics g = null;

        public MainForm()
        {
            InitializeComponent();
            g = this.CreateGraphics();

            DelayVolume.Text = Delay.Value.ToString();
            CountVolume.Text = Count.Value.ToString();

            MessageBox.Show("Для задания точек - кликайте левой кнопкой мыши.\nЗатем правой - для построения триангуляции.");
        }

        private void MainForm_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                if (points == null)
                {
                    points = new List<DelaunauTriangulationSample.Classes.Point>();
                    g.Clear(Color.White);
                }

                DelaunauTriangulationSample.Classes.Point mbToAdd = new DelaunauTriangulationSample.Classes.Point(e.X, Size.Height - e.Y);
                if (!points.Contains(mbToAdd))
                    points.Add(mbToAdd);
                g.DrawEllipse(new Pen(Color.Red), e.X - 1, e.Y - 1, 3, 3);
            }
            else
            {
                if (points == null)
                {
                    g.Clear(Color.White);
                    Random rnd = new Random();
                    points = new List<DelaunauTriangulationSample.Classes.Point>();
                    for (int i = 0; i < Count.Value; i++)
                    {
                        DelaunauTriangulationSample.Classes.Point tmp = new DelaunauTriangulationSample.Classes.Point(rnd.Next(Size.Width - 200) + 30, rnd.Next(Size.Height - 100) + 50);
                        if (!points.Contains(tmp))
                            points.Add(tmp);
                        g.DrawEllipse(defaultPointPen, (int)tmp.X - 1, Size.Height - (int)tmp.Y + 1, 3, 3);
                    }
                }
                Triangulation tb = new Triangulation(points, g, defaultLinePen, defaultPointPen, defaultNewLintPen, Size.Height, Delay.Value);
                points = null;
            }
        }

        private void MainForm_ResizeEnd(object sender, EventArgs e)
        {
            points = null;
            g = this.CreateGraphics();
            g.Clear(Color.White);
        }

        private void MainForm_Resize(object sender, EventArgs e)
        {
            points = null;
            g = this.CreateGraphics();
            g.Clear(Color.White);
        }

        private void Count_Scroll(object sender, EventArgs e)
        {
            CountVolume.Text = Count.Value.ToString();
        }

        private void Delay_Scroll(object sender, EventArgs e)
        {
            DelayVolume.Text = Delay.Value.ToString();
        }
    }
}
