#include "kmeans_machine.cpp"
#include "kmeans_trainer.cpp"

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_type_caster_base.hpp>

namespace py = pybind11;
using namespace bob::learn::em;


PYBIND11_PLUGIN(kmeans)
{
    xt::import_numpy();

    py::module m("kmeans", "KMeans Bindings!");

    py::class_<KMeansMachine>(m, "KMeansMachine")
    	.def(py::init<>())
    	.def(py::init<size_t, size_t>())
    	.def_property_readonly("means", &KMeansMachine::getMeans)
    ;

    py::class_<KMeansTrainer>(m, "KMeansTrainer")
    	.def(py::init<>())
    	.def("initialize", &KMeansTrainer::initialize<xt::pytensor<double, 2>>)
    	.def("e_step", &KMeansTrainer::eStep<xt::pytensor<double, 2>>)
    	.def("m_step", &KMeansTrainer::mStep<xt::pytensor<double, 2>>)
    	.def("getZeroethOrderStats", &KMeansTrainer::getZeroethOrderStats)
        .def("getFirstOrderStats", &KMeansTrainer::getFirstOrderStats)
    	.def("compute_likelihood", &KMeansTrainer::computeLikelihood)
    ;

    return m.ptr();
}