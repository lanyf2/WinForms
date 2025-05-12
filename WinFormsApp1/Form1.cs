using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.XImgProc.Segmentation;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Modules;
using static System.Reflection.Metadata.BlobBuilder;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using System.Drawing.Imaging;

namespace WinFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //create_template best_match 基于灰度值的模板匹配，计算模板图像与检测图像之间的像素灰度差值的绝对值总和（SAD方法）或者平方差总和（SSD方法）
            //create_ncc_model find_ncc_model 基于相关性的模板匹配，使用一种归一化的互相关匹配
            //create_scaled_shape_model  get_shape_model_contours find_scaled_shape_model 使用边缘特征定位物体，不适用于旋转和缩放比较大的情况
            //create_component_model  find_component_model  基于组件的模板匹配
            //create_local_deformable_model get_deformable_model_contours find_local_deformable_model  基于形变的模板匹配
            //create_uncalib_descriptor_model  find_uncalib_descriptor_model  基于描述符的模板匹配
            //matchTemplate 基于点的模板匹配

            using var src = new Mat("123.bmp", ImreadModes.Grayscale);
            //using var dst = new Mat();

            //Cv2.Canny(src, dst, 50, 200);
            //Cv2.Line(dst, 0, 100, 300, 500, OpenCvSharp.Scalar.Blue, 4);
            var bmp = src.ToBitmap();

            pBox1.ImgData = bmp;

            //tt();
            //MessageBox.Show("");
        }

        void tcv()
        {
            //var gray = new Mat();
            //Cv2.CvtColor(image, gray, ColorConversionCodes.BGRA2GRAY);
            //pBox1.ImgData = gray.ToBitmap();
            //var length = Math.Max(image.Size().Width, image.Size().Height);
            //var im2 = new Mat(length, length, MatType.CV_8UC3, new OpenCvSharp.Scalar(0, 0, 0));
            ////var im2 = new Mat(Mat.Zeros(length, length, MatType.CV_8UC3));
            //Cv2.HConcat(im2, image);
            //pBox1.ImgData = im2.ToBitmap();
            //return;
        }

        void tt()
        {
            var env = OrtEnv.Instance();
            var p = env.GetAvailableProviders();
            var v = env.GetVersionString();

            using var gpuSessionOptoins = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            InferenceSession session = new InferenceSession("./yolov7sim.onnx", gpuSessionOptoins);
            //InferenceSession session = new InferenceSession("./yolov7sim.onnx");

            var m = OrtMemoryInfo.DefaultInstance;
            var m2 = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

            Bitmap bmp = (Bitmap)Image.FromFile("image1.jpg");
            Bitmap destBitmap = new Bitmap(640, 640);
            var g = Graphics.FromImage(destBitmap);
            g.Clear(Color.Black);

            //设置画布的描绘质量         
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

            var sz = Math.Max(bmp.Width, bmp.Height);
            float w = bmp.Width;
            float h = bmp.Height;
            float scale = 1;
            if (w > h)
            {
                scale = w / 640;
                h = 640 / w * h;
                w = 640;
            }
            else
            {
                scale = h / 640;
                w = 640 / h * w;
                h = 640;
            }
            g.DrawImage(bmp, new RectangleF(0, 0, w, h));
            BitmapData data = destBitmap.LockBits(new Rectangle(0, 0, destBitmap.Width, destBitmap.Height), ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            System.Numerics.Tensors.DenseTensor<float> input = new(new[] { 1, 3, 640, 640 });

            byte[] b = new byte[3];
            nint srcPtr = data.Scan0;
            for (int y = 0; y < destBitmap.Height; y++)
            {
                var p0 = srcPtr + y * data.Stride;
                for (int x = 0; x < destBitmap.Width; x++)
                {
                    Marshal.Copy(p0, b, 0, 3);
                    input[0, 0, y, x] = b[2];
                    input[0, 1, y, x] = b[1];
                    input[0, 2, y, x] = b[0];
                    p0 += 3;
                }
            }
            destBitmap.UnlockBits(data);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                input.Buffer, new long[] { 1, 3, 640, 640 });
            float[] imageShape = { 640, 640 };
            using var imageShapeOrtValue = OrtValue.CreateTensorValueFromMemory(imageShape, new long[] { 1, 2 });
            //inputOrtValue.StringTensorSetElementAt("images", 0);//tensor: float32[1,3,640,640]
            var inputs = new Dictionary<string, OrtValue>
            {
                { "images", inputOrtValue }
            };
            //var inputs = new Dictionary<string, OrtValue>
            //{
            //    { "input_1", inputOrtValue },
            //    { "image_shape", imageShapeOrtValue }
            //};

            var runOptions = new RunOptions();
            var outputs = session.Run(runOptions, inputs, session.OutputNames);
            //Debug.Assert(outputs.Count > 0, "Expecting some output");

            var ct = outputs.Count;
            var lastOutput = outputs[outputs.Count - 1];

            var outputTypeInfo = lastOutput.GetTypeInfo();
            var sequenceTypeInfo = outputTypeInfo.TensorTypeAndShapeInfo;
            //var elementsNum = lastOutput.GetValueCount();
            //using var firstMap = lastOutput.GetValue(0, OrtAllocator.DefaultInstance);
            //var mapTypeInfo = firstMap.GetTypeInfo().MapTypeInfo;

            var minfo = lastOutput.GetTensorMemoryInfo();

            var boxesSpan = lastOutput.GetTensorDataAsSpan<float>();
            //var scoresSpan = results[1].GetTensorDataAsSpan<float>();
            //var indicesSpan = results[2].GetTensorDataAsSpan<int>();
        }
        void tt2()
        {
            //torch 2.0.1.1 CUDA 11.7
            //var dv = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            //var model = torch.jit.load("C:\\d\\ml\\yolov7-main\\new.pt");
            //var model = torch.load("C:\\d\\ml\\yolov7-main\\yolov7.pt");
            //var model = torch.load("C:\\d\\ml\\yolov7-main\\model.pt");
            //model.to(dv);
            //OpenCvSharp.Dnn.CvDnn.ReadTorchBlob
            //var x = torch.rand(5, 3);
            //var m = torch.jit.load("C:\\d\\ml\\yolov7-main\\weights\\yolov7.pt", dv);
            //var x = torch.rand(5, 3, null, dv);
            //label1.Text = x.ToString(TensorStringStyle.Julia);
            //label1.Text = torch.__version__;
            var info = Cv2.GetBuildInformation();

            // opencv 推理
            var net = OpenCvSharp.Dnn.Net.ReadNetFromONNX("yolov7sim.onnx")!;  // 加载训练好的识别模型
            net.SetPreferableBackend(OpenCvSharp.Dnn.Backend.CUDA);
            //var net = OpenCvSharp.Dnn.Net.ReadNetFromONNX("yolov7.onnx")!;  // 加载训练好的识别模型
            var image = Cv2.ImRead("image1.jpg");  // 读取图片
            var image2 = new Mat(640, 640, MatType.CV_8UC3, new OpenCvSharp.Scalar(0));  // 读取图片
            image.CopyTo(image2[new Rect(0, 0, image.Width, image.Height)]);
            pBox1.ImgData = image2.ToBitmap();


            var blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(image2, 0.003921568627451, new OpenCvSharp.Size(640, 640), swapRB: true);  // 由图片加载数据 这里还可以进行缩放、归一化等预处理
                                                                                                                                      //var blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(image, 0.003921568627451, image.Size(), new OpenCvSharp.Scalar(127.0, 127.0, 127.0)) / 0.5;  // 由图片加载数据 这里还可以进行缩放、归一化等预处理
            var r = new ResourcesTracker();
            //label1.Text = blob.Size().ToString();

            net.SetInput(blob);  // 设置模型输入
            var predict = net.Forward(); // 推理出结果

            //var result_mat_to_float = new Mat(8400, 84, MatType.CV_32F, predict.Data);
            //pBox1.ImgData = blob.ToBitmap();
            var tm = DateTime.Now;
            List<float[]> result = processing_result(predict);

            var names = new string[] { "person", "bicycle", "car", "motorcycle", "airplane"
                , "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign"
                , "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"
                , "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag"
                , "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat"
                , "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass"
                , "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange"
                , "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant"
                , "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone"
                , "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            //0,0,27,55 person tie cake
            for (int i = 0; i < result.Count; i++)
            {
                Cv2.Rectangle(image, new Rect((int)result[i][0], (int)result[i][1], (int)result[i][2], (int)result[i][3]), OpenCvSharp.Scalar.Blue, 1);
            }
            pBox1.ImgData = image.ToBitmap();
            //pBox1.ImgData = output.ToBitmap();

            //OpenCvSharp.Dnn.CvDnn.NMSBoxes();

            //label1.Text = x.ToString(TensorStringStyle.Julia);
        }
        List<float[]> processing_result(Mat result, float conf_thres = 0.25f, float nms_thres = 0.45f)
        {
            int nc = result.Size(4) - 5;//80
            //[x, y, w, h,conf,cls_conf]
            int min_wh = 10, max_wh = 4096;// (pixels) minimum and maximum box width and height
            int max_det = 300; // maximum number of detections per image
            int max_nms = 30000;// maximum number of boxes into torchvision.ops.nms()
            List<float[]> re = new List<float[]>();
            List<float[]> re2 = new List<float[]>();

            //int batch = result.Size(0);//1
            //int cls = result.Size(1);//3
            //int width = result.Size(2);//20
            //int height = result.Size(3);//20
            //int s4 = result.Size(4);//85

            int batch = result.Size(0);//1
            int cls = result.Size(1);//25200
            int cat = result.Size(2);//85
            //int height = result.Size(3);//34
            //int s4 = result.Size(4);//-3
            var dat = result.Data;
            List<Rect2d> boxes = new List<Rect2d>();
            List<float> scors = new List<float>();
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < cls; c++)
                {
                    float conf = result.At<float>(b, c, 4);
                    //float[] ff2 = new float[cat];
                    //for (int i = 0; i < cat; i++)
                    //{
                    //    ff2[i] = result.At<float>(b, c, i);
                    //}
                    if (conf > conf_thres)
                    {
                        float[] ff = new float[cat];
                        Marshal.Copy(dat + (b * cls * cat + c * cat) * 4, ff, 0, cat);
                        int maxid = GetMaxIndex(ff, 5);
                        float confmax = conf * ff[maxid];
                        if (confmax > conf_thres)
                        {
                            scors.Add(confmax);
                            boxes.Add(new Rect2d(ff[0] - ff[2] / 2, ff[1] - ff[3] / 2, ff[2], ff[3]));
                            re.Add(new float[] { ff[0] - ff[2] / 2, ff[1] - ff[3] / 2, ff[2], ff[3], confmax, maxid - 5 });
                        }
                    }
                }
            }

            OpenCvSharp.Dnn.CvDnn.NMSBoxes(boxes, scors, conf_thres, nms_thres, out int[] reIndc);
            for (int i = 0; i < reIndc.Length; i++)
            {
                re2.Add(re[reIndc[i]]);
            }
            return re2;
        }
        int GetMaxIndex(float[] ff, int startid = -1, int endid = -1)
        {
            if (endid < 0)
                endid = ff.Length;
            if (startid < 0)
                startid = 0;
            int id = startid;
            for (int i = startid; i < endid; i++)
            {
                if (ff[i] > ff[id])
                    id = i;
            }
            return id;
        }

        class Yolo : torch.nn.Module
        {
            int stride = 0;  // strides computed during build
            bool export = false;  // onnx export
            bool end2end = false;
            bool include_nms = false;
            bool concat = false;

            int nc;
            int no;
            long nl;
            torch.Tensor na;
            torch.Tensor grid;
            torch.Tensor a;
            torch.Tensor m;
            protected Yolo(string name, int nc = 80, torch.Tensor? anchors = null, torch.Tensor ch = null) : base(name)
            {
                //super(Detect, self).__init__()
                this.nc = nc;  //# number of classes
                no = nc + 5;  //# number of outputs per anchor
                nl = anchors.size(0); //# number of detection layers
                na = anchors[0].size();// 2  # number of anchors
                grid = torch.zeros(1) * nl; //# init grid
                //a = torch.tensor(anchors).@float().view(nl, -1, 2);
                this.register_buffer("anchors", a);//# shape(nl,na,2)
                //this.register_buffer("anchor_grid", a.clone().view(nl, 1, -1, 1, 1, 2)); //# shape(nl,1,na,1,1,2)
                //this.m = nn.ModuleList(nn.Conv2d(x, no * na, 1) for x in ch) ; //# output conv
            }
        }
    }
}