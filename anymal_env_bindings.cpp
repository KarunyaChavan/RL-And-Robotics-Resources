/**
 * Python bindings for ANYmal blind_locomotion environment
 * Uses pybind11 to expose C++ environment to Python for training
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

// Only include c100 to avoid duplicate definitions
#include <environment/environment_c100.hpp>

namespace py = pybind11;

// Helper to convert Eigen matrices to numpy arrays
template<typename Derived>
py::array_t<typename Derived::Scalar> eigen_to_numpy(const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;
    py::array_t<Scalar> result({mat.rows(), mat.cols()});
    auto buf = result.request();
    Scalar* ptr = static_cast<Scalar*>(buf.ptr);
    
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            ptr[i * mat.cols() + j] = mat(i, j);
        }
    }
    return result;
}

PYBIND11_MODULE(anymal_env, m) {
    m.doc() = "Python bindings for ANYmal blind locomotion environment";
    
    // Expose enums
    py::enum_<Env::ActionType>(m, "ActionType")
        .value("EE", Env::ActionType::EE)
        .value("JOINT", Env::ActionType::JOINT)
        .export_values();
    
    py::enum_<Env::CommandMode>(m, "CommandMode")
        .value("RANDOM", Env::CommandMode::RANDOM)
        .value("FIXED_DIR", Env::CommandMode::FIXED_DIR)
        .value("STRAIGHT", Env::CommandMode::STRAIGHT)
        .value("STOP", Env::CommandMode::STOP)
        .value("ZERO", Env::CommandMode::ZERO)
        .value("NOZERO", Env::CommandMode::NOZERO)
        .export_values();

    // Bind the c100 version
    py::class_<Env::blind_locomotion>(m, "BlindLocomotionC100")
        .def(py::init<bool, int, std::string, std::string>(),
             py::arg("visualize") = false,
             py::arg("instance") = 0,
             py::arg("urdf_path") = "",
             py::arg("actuator_path") = "",
             "Initialize ANYmal C100 environment")
        
        // Environment control
        .def("init", &Env::blind_locomotion::init,
             "Initialize the environment")
        
        // Note: There is no separate reset() method - use init() to reset
        
        .def("integrate", &Env::blind_locomotion::integrate,
             "Integrate physics for one control step (50Hz)")
        
        // Terrain and task
        .def("updateTask", [](Env::blind_locomotion& self, py::array_t<float> params) {
            auto buf = params.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Task params must be 1-dimensional");
            }
            Eigen::Matrix<float, -1, 1> task_params(buf.shape[0]);
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; ++i) {
                task_params[i] = ptr[i];
            }
            self.updateTask(task_params);
        }, py::arg("task_params"),
           "Update terrain parameters: [task_idx, param1, param2, param3]")
        
        .def("setFootFriction", &Env::blind_locomotion::setFootFriction,
             py::arg("foot_id"), py::arg("friction"),
             "Set friction coefficient for a specific foot")
        
        // State observation (for student/proprioceptive)
        .def("getState", [](Env::blind_locomotion& self) {
            // Use dynamic-size matrix to match C++ signature
            Eigen::Matrix<float, -1, 1> state;
            self.getState(state);
            return eigen_to_numpy(state);
        }, "Get current state (133-dim)")
        
        // Note: getObservation doesn't exist as a separate method in C++
        // The observation is the most recent column of the history buffer
        // Use getHistory with history_length=1 to get current observation
        .def("getObservation", [](Env::blind_locomotion& self) {
            Eigen::Matrix<float, -1, -1> history;
            self.getHistory(history, 1);
            // Return first column as 1D array
            Eigen::Matrix<float, Env::ObservationDim, 1> obs = history.col(0);
            return eigen_to_numpy(obs);
        }, "Get current observation (60-dim)")
        
        // History for TCN student
        .def("getHistory", [](Env::blind_locomotion& self, int history_len) {
            Eigen::Matrix<float, -1, -1> history;
            self.getHistory(history, history_len);
            return eigen_to_numpy(history);
        }, py::arg("history_length"),
           "Get observation history for TCN (60 x history_length)")
        
        // Privileged state (for teacher)
        .def("getPrivilegedState", [](Env::blind_locomotion& self) {
            Eigen::Matrix<float, -1, -1> priv_state;
            self.getPriviligedState(priv_state);
            // Return as 1D array (column vector)
            py::array_t<float> result({priv_state.rows()});
            auto buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < priv_state.rows(); ++i) {
                ptr[i] = priv_state(i, 0);
            }
            return result;
        }, "Get privileged state with terrain info (230-dim)")
        
        // Action
        .def("updateAction", [](Env::blind_locomotion& self, py::array_t<float> action) {
            auto buf = action.request();
            if (buf.ndim != 1 || buf.shape[0] != Env::ActionDim) {
                throw std::runtime_error("Action must be " + std::to_string(Env::ActionDim) + "-dimensional");
            }
            Eigen::Matrix<float, Env::ActionDim, 1> act;
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < Env::ActionDim; ++i) {
                act[i] = ptr[i];
            }
            self.updateAction(act);
        }, py::arg("action"),
           "Apply action (16-dim: 4 frequencies + 12 foot position residuals)")
        
        // Note: The following methods don't exist in the C++ class:
        // - getReward, isTerminal, getSimulationTime, getControlTime
        // These would need to be implemented in Python if needed
        
        // Command setting
        .def("sampleCommand", &Env::blind_locomotion::sampleCommand,
             "Sample a new random locomotion command")
        
        .def("updateCommand", &Env::blind_locomotion::updateCommand,
             "Update command based on goal position")
        
        // Utility
        .def("seed", &Env::blind_locomotion::seed,
             py::arg("seed"),
             "Set random seed")
        
        // Additional methods for training
        .def_property_readonly("steps", [](Env::blind_locomotion& self) {
            return self.steps_;
        }, "Get current step count")
        
        .def_property("command",
            [](Env::blind_locomotion& self) {
                py::array_t<double> cmd(3);
                auto buf = cmd.request();
                double* ptr = static_cast<double*>(buf.ptr);
                for(int i = 0; i < 3; i++) ptr[i] = self.command_[i];
                return cmd;
            },
            [](Env::blind_locomotion& self, py::array_t<double> cmd) {
                auto buf = cmd.request();
                if (buf.ndim != 1 || buf.shape[0] != 3) {
                    throw std::runtime_error("Command must be 3-dimensional");
                }
                double* ptr = static_cast<double*>(buf.ptr);
                for(int i = 0; i < 3; i++) self.command_[i] = ptr[i];
            }, "Get/set locomotion command [vx, vy, omega]")
        
        .def("getPhases", [](Env::blind_locomotion& self) {
            py::array_t<double> phases(4);
            auto buf = phases.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for(int i = 0; i < 4; i++) ptr[i] = self.pi_[i];
            return phases;
        }, "Get CPG phases for each leg")
        
        .def("getPhaseDerivatives", [](Env::blind_locomotion& self) {
            py::array_t<double> phase_d(4);
            auto buf = phase_d.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for(int i = 0; i < 4; i++) ptr[i] = self.piD_[i];
            return phase_d;
        }, "Get CPG phase derivatives for each leg")
        
        .def("getContactState", [](Env::blind_locomotion& self) {
            py::array_t<bool> contacts(4);
            auto buf = contacts.request();
            bool* ptr = static_cast<bool*>(buf.ptr);
            for(int i = 0; i < 4; i++) ptr[i] = self.footContactState_[i];
            return contacts;
        }, "Get foot contact states (True if in contact)")
        
        .def("getContactCounts", [](Env::blind_locomotion& self) {
            return py::make_tuple(
                self.numContact_,
                self.numFootContact_,
                self.numBaseContact_,
                self.numShankContact_,
                self.numThighContact_
            );
        }, "Get contact counts (total, foot, base, shank, thigh)")
        
        .def("isBadlyConditioned", [](Env::blind_locomotion& self) {
            return self.badlyConditioned_;
        }, "Check if simulation is badly conditioned (NaN/Inf)")
        
        .def("isTerminal", [](Env::blind_locomotion& self) {
            return self.badlyConditioned_ || self.numBaseContact_ > 0;
        }, "Check if episode should terminate")
        
        .def("getBaseVelocity", [](Env::blind_locomotion& self) {
            Eigen::VectorXd base_vel = self.getBaseVelocity();
            return eigen_to_numpy(base_vel);
        }, "Get base velocity in body frame [vx, vy, vz, wx, wy, wz]")
        
        .def("getGeneralizedState", [](Env::blind_locomotion& self) {
            Eigen::VectorXd q = self.getGeneralizedState();
            return eigen_to_numpy(q);
        }, "Get generalized coordinates (19-dim)")
        
        .def("getGeneralizedVelocity", [](Env::blind_locomotion& self) {
            Eigen::VectorXd u = self.getGeneralizedVelocity();
            return eigen_to_numpy(u);
        }, "Get generalized velocities (18-dim)")
        
        // Configuration methods
        .def("setActionType", &Env::blind_locomotion::setActionType,
             py::arg("action_type"),
             "Set action type (EE or JOINT)")
        
        .def("setCommandMode", &Env::blind_locomotion::setCommandMode,
             py::arg("command_mode"),
             "Set command mode (RANDOM, FIXED_DIR, STRAIGHT, STOP, ZERO, NOZERO)")
        
        .def("setRealTimeFactor", &Env::blind_locomotion::setRealTimeFactor,
             py::arg("factor"),
             "Set real-time visualization speed factor")
        
        .def("addBaseMass", &Env::blind_locomotion::addBaseMass,
             py::arg("mass"),
             "Add mass to base for domain randomization")
        
        // Video recording
        .def("startRecordingVideo", &Env::blind_locomotion::startRecordingVideo,
             py::arg("path"),
             "Start recording video to specified path")
        
        .def("endRecordingVideo", &Env::blind_locomotion::endRecordingVideo,
             "Stop recording video and save");
    
    // Constants
    m.attr("STATE_DIM") = Env::StateDim;
    m.attr("OBS_DIM") = Env::ObservationDim;
    m.attr("ACTION_DIM") = Env::ActionDim;
    m.attr("PRIVILEGED_STATE_DIM") = Env::PrivilegedStateDim;
}
