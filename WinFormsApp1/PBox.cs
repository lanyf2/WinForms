using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WinFormsApp1
{
    public partial class PBox : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        private void NotifyPropertyChanged(string name)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }

        private double _dScale = 1.0;
        public double DScale
        {
            get { return _dScale; }
            set
            {
                if (value < 0)
                {
                    return;
                }
                else if (value == 0)
                {
                    _dScale = 1;
                }
                else
                {
                    _dScale = value;
                }
                NotifyPropertyChanged("DScale");
            }
        }
        private Matrix _matrix = new();

        private Rectangle _rcImg = new Rectangle(0, 0, 0, 0);
        private Bitmap? _bitmap;
        public Bitmap? ImgData
        {
            set
            {
                InitPara();
                _bitmap = value;
                if (_bitmap == null)
                {
                    _rcImg.Width = 0;
                    _rcImg.Height = 0;
                }
                else
                {
                    _rcImg.Width = _bitmap.Width;
                    _rcImg.Height = _bitmap.Height;
                    AdaptView();
                    TranslationCenter();

                }
                Refresh();
            }
            get { return _bitmap; }
        }

        private Rectangle rcBgArea = new();    // 图片背景区域
        //private Brush _ImgBg = new SolidBrush(Color.FromArgb(0xA8, 0xA8, 0xA8));     // 图片颜色背景

        private Point _LastPt = new();
        public PBox()
        {
            InitializeComponent();
            SetStyle(ControlStyles.AllPaintingInWmPaint, true);
            SetStyle(ControlStyles.UserPaint, true);
            SetStyle(ControlStyles.OptimizedDoubleBuffer, true);
            InitPara();
        }
        private void InitPara()
        {
            _rcImg = new Rectangle(0, 0, 0, 0);
            _matrix.Reset();
            _LastPt = new();
        }
        private void ChartView_Load(object sender, EventArgs e)
        {
            this.CalculationArea();
        }

        private void ChartView_Resize(object sender, EventArgs e)
        {
            this.CalculationArea();
        }
        private void PBox_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            AdaptView();
            TranslationCenter();
        }

        private void CalculationArea()
        {
            // 控件大小
            int nW = this.Width;
            int nH = this.Height;


            // 图片区域
            rcBgArea.X = 0;
            rcBgArea.Y = 0;
            rcBgArea.Width = nW;
            rcBgArea.Height = nH;

        }
        private void ChartView_Paint(object sender, PaintEventArgs e)
        {
            Graphics graph = e.Graphics;
            DrawView(graph);

        }
        private void ChartView_Paint_bak(object sender, PaintEventArgs e)
        {
            Graphics graph = e.Graphics;
            graph.SmoothingMode = SmoothingMode.HighQuality;
            // 双缓冲绘图
            Bitmap bmpChartView = new(Width, Height);
            Graphics bmpChartView_g = Graphics.FromImage(bmpChartView);
            bmpChartView_g.SmoothingMode = SmoothingMode.HighQuality;

            DrawView(bmpChartView_g);
            graph.DrawImage(bmpChartView, 0, 0);

            bmpChartView_g.Dispose();
            bmpChartView.Dispose();
        }
        private void DrawView(Graphics graph)
        {
            DrawMainView(graph);
        }
        private void DrawMainView(Graphics graph)
        {
            graph.FillRectangle(new SolidBrush(BackColor), rcBgArea);
            if (_bitmap == null)
                return;
            graph.Transform = _matrix;
            graph.DrawImage(_bitmap, 0, 0, _bitmap.Width, _bitmap.Height);

        }
        private void DrawMainViewbak(Graphics graph)
        {
            // 填充背景
            graph.FillRectangle(new SolidBrush(BackColor), rcBgArea);
            if (_bitmap == null)
            {
                return;
            }

            Bitmap bitImg = new Bitmap(rcBgArea.Width, rcBgArea.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            Graphics bitImg_g = Graphics.FromImage(bitImg);
            bitImg_g.Transform = _matrix;
            bitImg_g.DrawImage(_bitmap, 0, 0, _bitmap.Width, _bitmap.Height);

            graph.DrawImage(bitImg, rcBgArea.Left, rcBgArea.Top, bitImg.Width, bitImg.Height);

            //foreach (var chart in LstChart)
            //{
            //    chart.DrawChart(graph, _matrix);
            //}

            bitImg_g.Dispose();
            bitImg.Dispose();
        }
        private void ChartView_MouseWheel(object sender, MouseEventArgs e)
        {
            Point[] points = new Point[] { e.Location };
            Matrix matrix_Invert = _matrix.Clone();
            matrix_Invert.Invert();
            matrix_Invert.TransformPoints(points);
            //Console.WriteLine(points[0]);
            //if (_rcImg.Contains(points[0]))
            {
                double step = 1.3;
                if (e.Delta < 0)
                {
                    step = 1.0 / 1.3;
                }
                DScale *= step;
                _matrix.Scale((float)step, (float)step);

                Point[] pointse = new Point[] { e.Location };
                matrix_Invert = _matrix.Clone();
                matrix_Invert.Invert();
                matrix_Invert.TransformPoints(pointse);
                _matrix.Translate((pointse[0].X - points[0].X), (pointse[0].Y - points[0].Y));

                Refresh();
            }
        }
        public void AdaptView()
        {
            if (_bitmap == null)
            {
                return;
            }

            using GraphicsPath graphPath = new();
            graphPath.AddRectangle(_rcImg);
            graphPath.Transform(_matrix);
            PointF[] pointFs = graphPath.PathPoints;
            float fxmin = pointFs[0].X;
            float fymin = pointFs[0].Y;
            float fxmax = pointFs[0].X;
            float fymax = pointFs[0].Y;

            foreach (var pt in pointFs)
            {
                if (pt.X < fxmin)
                {
                    fxmin = pt.X;
                }
                else if (pt.X > fxmax)
                {
                    fxmax = pt.X;
                }
                if (pt.Y < fymin)
                {
                    fymin = pt.Y;
                }
                else if (pt.Y > fymax)
                {
                    fymax = pt.Y;
                }
            }

            float fWidth = fxmax - fxmin;
            float fHeight = fymax - fymin;

            if (fWidth * rcBgArea.Height < fHeight * rcBgArea.Width)
            {
                DScale = rcBgArea.Height / fHeight;
            }
            else
            {
                DScale = rcBgArea.Width / fWidth;
            }
            _matrix.Scale((float)DScale, (float)DScale, MatrixOrder.Append);
        }
        public void TranslationCenter()
        {
            if (_bitmap == null)
            {
                return;
            }
            Matrix matrixinv = _matrix.Clone();
            matrixinv.Invert();

            Point[] ptViewCenter = new Point[] { new Point(rcBgArea.Left + rcBgArea.Width / 2, rcBgArea.Top + rcBgArea.Height / 2) };
            matrixinv.TransformPoints(ptViewCenter);

            _matrix.Translate(ptViewCenter[0].X - _rcImg.Width / 2, ptViewCenter[0].Y - _rcImg.Height / 2);
            this.Refresh();

        }
    }
}
