/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */
#ifndef BOB_LEARN_EM_KMEANSMACHINE_H
#define BOB_LEARN_EM_KMEANSMACHINE_H

#include <xtensor/xtensor.hpp>
#include <xtensor/xexpression.hpp>
#include <cfloat>

// #include <bob.io.base/HDF5File.h>

namespace bob { namespace learn { namespace em {

/**
 * @brief This class implements a k-means classifier.
 * @details See Section 9.1 of Bishop, "Pattern recognition and machine learning", 2006
 */
class KMeansMachine {
  public:
    /**
     * Default constructor. Builds an otherwise invalid 0 x 0 k-means
     * machine. This is equivalent to construct a LinearMachine with two
     * size_t parameters set to 0, as in LinearMachine(0, 0).
     */
    KMeansMachine();

    /**
     * Constructor
     * @param[in] n_means  The number of means
     * @param[in] n_inputs The feature dimensionality
     */
    KMeansMachine(const size_t n_means, const size_t n_inputs);

    /**
     * Builds a new machine with the given means. Each row of the means
     * matrix should represent a mean.
     */
    KMeansMachine(const xt::xtensor<double, 2>& means);

    /**
     * Copies another machine (copy constructor)
     */
    KMeansMachine(const KMeansMachine& other);

    /**
     * Starts a new KMeansMachine from an existing Configuration object.
     */
    // KMeansMachine(bob::io::base::HDF5File& config);

    /**
     * Destructor
     */
    virtual ~KMeansMachine();

    /**
     * Assigns from a different machine
     */
    KMeansMachine& operator=(const KMeansMachine& other);

    /**
     * Equal to
     */
    bool operator==(const KMeansMachine& b) const;

    /**
     * Not equal to
     */
    bool operator!=(const KMeansMachine& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const KMeansMachine& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * Loads data from an existing configuration object. Resets the current
     * state.
     */
    // void load(bob::io::base::HDF5File& config);

    /**
     * Saves an existing machine to a Configuration object.
     */
    // void save(bob::io::base::HDF5File& config) const;

    /**
     * Output the minimum (Square Euclidean) distance between the input and
     * one of the means (overrides Machine::forward)
     */
    void forward(const xt::xtensor<double, 1>& input, double& output) const;

    /**
     * Output the minimum (Square Euclidean) distance between the input and
     * one of the means (overrides Machine::forward_)
     * @warning Inputs are NOT checked
     */
    void forward_(const xt::xtensor<double, 1>& input, double& output) const;


    /**
     * Set the means
     */
    void setMeans(const xt::xtensor<double, 2>& means);

    /**
     * Set the i'th mean
     */
    void setMean(const size_t i, const xt::xtensor<double,1>& mean);

    /**
     * Get a mean
     * @param[in]   i    The index of the mean
     * @param[out] mean The mean, a 1D array, with a length equal to the number of feature dimensions.
     */
    const xt::xtensor<double, 1> getMean(const size_t i) const;

    /**
     * Get the means (i.e. a 2D array, with as many rows as means, and as
     * many columns as feature dimensions.)
     */
    const xt::xtensor<double, 2>& getMeans() const
    { return m_means; }

     /**
     * Get the means in order to be updated (i.e. a 2D array, with as many
     * rows as means, and as many columns as feature dimensions.)
     * @warning Only trainers should use this function for efficiency reasons
     */
    xt::xtensor<double,2>& updateMeans()
    { return m_means; }

    /**
     * Return the power of two of the (Square Euclidean) distance of the
     * sample, x, to the i'th mean
     * @param x The data sample (feature vector)
     * @param i The index of the mean
     */
    double getDistanceFromMean(const xt::xtensor<double,1>& x,
      const size_t i) const;

    /**
     * Calculate the index of the mean that is closest
     * (in terms of Square Euclidean distance) to the data sample, x
     * @param x The data sample (feature vector)
     * @param closest_mean (output) The index of the mean closest to the sample
     * @param min_distance (output) The distance of the sample from the closest mean
     */
    void getClosestMean(const xt::xtensor<double,1>& x,
      size_t &closest_mean, double &min_distance) const;

    /**
     * Output the minimum (Square Euclidean) distance between the input and
     * one of the means
     */
    double getMinDistance(const xt::xtensor<double,1>& input) const;

    /**
     * For each mean, find the subset of the samples
     * that is closest to that mean, and calculate
     * 1) the variance of that subset (the cluster variance)
     * 2) the proportion of the samples represented by that subset (the cluster weight)
     * @param[in]  data      The data
     * @param[out] variances The cluster variances (one row per cluster),
     *                       with as many columns as feature dimensions.
     * @param[out] weights   A vector of weights, one per cluster
     */
    void getVariancesAndWeightsForEachCluster(const xt::xtensor<double,2> &data, xt::xtensor<double,2>& variances, xt::xtensor<double,1>& weights) const;
    /**
     * Methods consecutively called by getVariancesAndWeightsForEachCluster()
     * This should help for the parallelization on several nodes by splitting the data and calling
     * getVariancesAndWeightsForEachClusterAcc() for each split. In this case, there is a need to sum
     * with the m_cache_means, variances, and weights variables before performing the merge on one
     * node using getVariancesAndWeightsForEachClusterFin().
     */
    void getVariancesAndWeightsForEachClusterInit(xt::xtensor<double,2>& variances, xt::xtensor<double,1>& weights) const;
    void getVariancesAndWeightsForEachClusterAcc(const xt::xtensor<double,2> &data, xt::xtensor<double,2>& variances, xt::xtensor<double,1>& weights) const;
    void getVariancesAndWeightsForEachClusterFin(xt::xtensor<double,2>& variances, xt::xtensor<double,1>& weights) const;

    /**
     * Get the m_cache_means array.
     * @warning This variable should only be used in the case you want to parallelize the
     * getVariancesAndWeightsForEachCluster() method!
     */
    const xt::xtensor<double, 2>& getCacheMeans() const
    { return m_cache_means; }

    /**
     * Set the m_cache_means array.
     * @warning This variable should only be used in the case you want to parallelize the
     * getVariancesAndWeightsForEachCluster() method!
     */
    void setCacheMeans(const xt::xtensor<double, 2>& cache_means);

    /**
     * Resize the means
     */
    void resize(const size_t n_means, const size_t n_inputs);

    /**
     * Return the number of means
     */
    size_t getNMeans() const { return m_n_means; }

    /**
     * Return the number of inputs
     */
    size_t getNInputs() const { return m_n_inputs; }

    /**
     * Prints a KMeansMachine in the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const KMeansMachine& km);


  private:
     /**
     * The number of means
     */
    size_t m_n_means;

    /**
     * The number of inputs
     */
    size_t m_n_inputs;

    /**
     * The means (each row is a mean)
     */
    xt::xtensor<double, 2> m_means;

    /**
     * cache to avoid re-allocation
     */
    mutable xt::xtensor<double, 2> m_cache_means;
};

} } } // namespaces

#endif // BOB_LEARN_EM_KMEANSMACHINE_H
