#include <systemc>
#include <iostream>

using namespace sc_core;
using namespace sc_dt;
using namespace std;

// ====== Global configuration (mirroring embedded_app.py logic) ======

// Camera config (can be 640x480 or 5MP 2592x1944)
static const int CAM_WIDTH  = 640;    // or 2592 for 5MP
static const int CAM_HEIGHT = 480;    // or 1944 for 5MP

// Target camera capture FPS (pygame clock.tick(30) ~ 30 fps)
static const double CAM_INPUT_FPS = 30.0;

// Approximate OpenCV preprocessing cost (grayscale + Haar + resize)
// -> you can calibrate this using timing results from Python
static const double PREPROCESS_MS = 10.0;  // e.g. 10 ms per frame

// Approximate TFLite CNN latency on Raspberry Pi 4 (from benchmark_log)
// If you measured avg_inference_ms ~ 120 ms on Pi 4, use that here
static const double INFERENCE_MS  = 120.0; // ~7–8 FPS bottleneck

// Time for DB lookup + audio trigger (very small compared to CNN)
static const double DB_AUDIO_MS   = 2.0;

// Total virtual simulation time
static const double SIM_SECONDS   = 3.0;   // 3 seconds like scan_duration in Python


// ====== Frame structure (simplified) ======
struct Frame {
    int   id;
    int   width;
    int   height;
    bool  face_detected;  // just a flag; we don't simulate detection logic
};

// For optional debug printing
std::ostream& operator<<(std::ostream& os, const Frame& f) {
    os << "Frame{id=" << f.id
       << ", w=" << f.width
       << ", h=" << f.height
       << ", face=" << (f.face_detected ? "true" : "false") << "}";
    return os;
}


// ====== Camera module (simulates cv2.VideoCapture + cap.read) ======
SC_MODULE(Camera) {
    sc_fifo_out<Frame> out;
    sc_time frame_period; // time between two captured frames

    SC_CTOR(Camera)
        : frame_period( sc_time( 1000.0 / CAM_INPUT_FPS, SC_MS ) ) // e.g. ~33.3 ms for 30 FPS
    {
        SC_THREAD(capture_loop);
    }

    void capture_loop() {
        int id = 0;
        while (true) {
            Frame f;
            f.id = id++;
            f.width  = CAM_WIDTH;
            f.height = CAM_HEIGHT;
            f.face_detected = true; // assume at least one face; we only care about timing

            out.write(f);
            wait(frame_period); // simulate camera capture interval
        }
    }
};


// ====== OpenCV Preprocess module ======
// Models: cvtColor -> detectMultiScale -> crop -> resize -> normalize
SC_MODULE(OpenCVPreprocess) {
    sc_fifo_in<Frame>  in;
    sc_fifo_out<Frame> out;
    sc_time preprocess_delay;

    SC_CTOR(OpenCVPreprocess)
        : preprocess_delay( sc_time(PREPROCESS_MS, SC_MS) )
    {
        SC_THREAD(process_loop);
    }

    void process_loop() {
        while (true) {
            Frame f = in.read();

            // Here in Python you do:
            // gray = cv2.cvtColor(frame, COLOR_BGR2GRAY)
            // faces = face_cascade.detectMultiScale(...)
            // roi_gray = gray[y:y+h, x:x+w]
            // img_resized = cv2.resize(roi_gray, (width,height))
            // normalize / expand dims / etc.
            //
            // In SystemC, we only simulate the total time cost:
            wait(preprocess_delay);

            out.write(f);
        }
    }
};


// ====== TFLite Inference module ======
// Models: interpreter.set_tensor -> interpreter.invoke -> get_tensor
SC_MODULE(TFLiteInference) {
    sc_fifo_in<Frame>  in;
    sc_fifo_out<Frame> out;
    sc_time inference_delay;

    SC_CTOR(TFLiteInference)
        : inference_delay( sc_time(INFERENCE_MS, SC_MS) )
    {
        SC_THREAD(inference_loop);
    }

    void inference_loop() {
        while (true) {
            Frame f = in.read();

            // In Python you measure:
            // t0 = time.time()
            // interpreter.set_tensor(...)
            // interpreter.invoke()
            // t1 = time.time()
            // avg_inference_ms = ...
            //
            // In SystemC we wait for that avg time:
            wait(inference_delay);

            out.write(f);
        }
    }
};


// ====== Music Recommender / DB module ======
// Models: SQLite query + song list retrieval
SC_MODULE(MusicRecommender) {
    sc_fifo_in<Frame>  in;
    sc_fifo_out<Frame> out;
    sc_time db_delay;

    SC_CTOR(MusicRecommender)
        : db_delay( sc_time(DB_AUDIO_MS, SC_MS) )
    {
        SC_THREAD(recommend_loop);
    }

    void recommend_loop() {
        while (true) {
            Frame f = in.read();

            // In Python: get_recommendations_from_db(processed_emotions)
            // In SystemC: just simulate DB + minor logic
            wait(db_delay);

            out.write(f);
        }
    }
};


// ====== Sink / Benchmark module ======
// This approximates the benchmark_results dict in Python
SC_MODULE(SinkBenchmark) {
    sc_fifo_in<Frame> in;

    int      frame_count;
    sc_time  first_time;
    sc_time  last_time;

    SC_CTOR(SinkBenchmark)
        : frame_count(0)
    {
        SC_THREAD(collect_loop);
    }

    void collect_loop() {
        first_time = SC_ZERO_TIME;
        last_time  = SC_ZERO_TIME;

        while (true) {
            Frame f = in.read();
            sc_time now = sc_time_stamp();

            if (frame_count == 0) {
                first_time = now;
            }

            last_time = now;
            frame_count++;

            // Print every 5 or 10 frames
            if (frame_count % 5 == 0) {
                double sim_time_ms = last_time.to_seconds() * 1000.0;
                double effective_fps =
                    frame_count / (last_time - first_time).to_seconds();

                cout << "[BENCH] t=" << sim_time_ms << " ms"
                     << " | frames=" << frame_count
                     << " | effective FPS=" << effective_fps
                     << endl;
            }
        }
    }
};


// ====== Top-level sc_main ======
int sc_main(int argc, char* argv[]) {
    // FIFO channels between modules
    sc_fifo<Frame> fifo_cam2pre(10);
    sc_fifo<Frame> fifo_pre2inf(10);
    sc_fifo<Frame> fifo_inf2db(10);
    sc_fifo<Frame> fifo_db2sink(10);

    // Instantiate modules
    Camera           cam("Camera");
    OpenCVPreprocess pre("OpenCVPreprocess");
    TFLiteInference  inf("TFLiteInference");
    MusicRecommender rec("MusicRecommender");
    SinkBenchmark    sink("SinkBenchmark");

    // Connect
    cam.out(fifo_cam2pre);
    pre.in(fifo_cam2pre);
    pre.out(fifo_pre2inf);
    inf.in(fifo_pre2inf);
    inf.out(fifo_inf2db);
    rec.in(fifo_inf2db);
    rec.out(fifo_db2sink);
    sink.in(fifo_db2sink);

    cout << "Starting SystemC simulation for "
         << SIM_SECONDS << " simulated seconds..." << endl;

    sc_start( SIM_SECONDS, SC_SEC );

    cout << "Simulation finished at " << sc_time_stamp() << endl;

    return 0;
}