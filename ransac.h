#include <random>
#include <iomanip>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <vector>

struct Plane
{
    double a;
    double b;
    double c;
    double d;
};

struct FitResult
{
    Plane plane;
    int n_inliers = -1;
};

void pretty_print(std::string const &input_filename, Plane const &plane)
{
    std::cout << std::setprecision(3) << "--infile " << input_filename << " -a " << plane.a << " -b " << plane.b << " -c " << plane.c << " -d " << plane.d << '\n';
}

std::ostream &operator<<(std::ostream &os, Plane const &plane)
{
    os << std::setprecision(3) << plane.a << " " << plane.b << " " << plane.c << " " << plane.d;
    return os;
}

Plane compute_plane_from_points(Eigen::Vector3d const &p0, Eigen::Vector3d const &p1, Eigen::Vector3d const &p2)
{
    Eigen::Vector3d vect1;
    Eigen::Vector3d vect2;
    vect1 = p1 - p0;
    vect2 = p2 - p0;
    Eigen::VectorXd normal_vec = vect1.cross(vect2);
    Plane p_plane;
    p_plane.a = normal_vec(0);
    p_plane.b = normal_vec(1);
    p_plane.c = normal_vec(2);
    p_plane.d = -normal_vec.dot(p0);
    return p_plane;
}

class BaseFitter
{
public:
    BaseFitter(int num_points) : mt(rd()), dist(0, num_points - 1)
    {
        mt.seed(0);
    }

    /**
     * Given all of the data `points`, select a random subset and fit a plane to that subset.
     * the parameter points is all of the points
     * the return value is the FitResult which contains the parameters of the plane (a,b,c,d) and the number of inliers
     */
    virtual FitResult fit(Eigen::MatrixXd const &points) = 0;

    int get_random_point_idx()
    {
        return dist(mt);
    };

    double const inlier_threshold_{0.09};

private:
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist;
};

class AnalyticFitter : public BaseFitter
{
public:
    AnalyticFitter(int num_points) : BaseFitter(num_points) {}

    FitResult fit(Eigen::MatrixXd const &points) override
    {

        int random_point1 = get_random_point_idx();
        int random_point2 = get_random_point_idx();
        int random_point3 = get_random_point_idx();

        Eigen::Vector3d rand_vec1 = points.row(random_point1);
        Eigen::Vector3d rand_vec2 = points.row(random_point2);
        Eigen::Vector3d rand_vec3 = points.row(random_point3);

        Plane analytic_plane = compute_plane_from_points(rand_vec1, rand_vec2, rand_vec3);
        int count = 0;
        for (int i = 0; i < points.rows(); i++)
        {
            double numerator = abs((analytic_plane.a * points(i, 0)) + (analytic_plane.b * points(i, 1)) + (analytic_plane.c * points(i, 2)) + (analytic_plane.d));
            double denom = sqrt((analytic_plane.a * analytic_plane.a) + (analytic_plane.b * analytic_plane.b) + (analytic_plane.c * analytic_plane.c));
            double val = numerator / denom;

            if (val <= inlier_threshold_)
            {
                count += 1;
            }
        }
        int n_inliers = count;

        return {analytic_plane, n_inliers};
    }
};

class LeastSquaresFitter : public BaseFitter
{
public:
    LeastSquaresFitter(int num_points, int n_sample_points) : BaseFitter(num_points), n_sample_points_(n_sample_points) {}

    FitResult fit(Eigen::MatrixXd const &points) override
    {
        Eigen::MatrixXd ran_les(n_sample_points_,3);
        for (int i = 0; i < n_sample_points_; i++)
        {
            int var = get_random_point_idx();
            ran_les.row(i) = points.row(var);
        }

        Eigen::MatrixXd A_matrix(n_sample_points_, 3);
        A_matrix = ran_les.block(0, 0, n_sample_points_, 2).rowwise().homogeneous();

        Eigen::VectorXd b_matrix(n_sample_points_);
        b_matrix = ran_les.col(2);
        Eigen::Vector3d x_var = A_matrix.colPivHouseholderQr().solve(b_matrix);
        Plane a_var;
        a_var.a = x_var(0);
        a_var.b = x_var(1);
        a_var.c = -1;
        a_var.d = x_var(2);

        int count = 0;
        for (int i = 0; i < points.rows(); i++)
        {
            double numerator = abs((a_var.a * points(i, 0)) + (a_var.b * points(i, 1)) + (a_var.c * points(i, 2)) + (a_var.d));
            double denom = sqrt((a_var.a * a_var.a) + (a_var.b * a_var.b) + (a_var.c * a_var.c));
            double val = numerator / denom;

            if (val <= inlier_threshold_)
            {
                count += 1;
            }
        }
        int n_inliers = count;

        return {a_var,n_inliers};
    }
    int const n_sample_points_;
};

Plane ransac(BaseFitter &fitter, Eigen::MatrixXd const &points)
{

    FitResult best_result;
    best_result = fitter.fit(points);
    for (int i = 0; i<=25; i++)
    {
        FitResult result = fitter.fit(points);
        if(result.n_inliers> best_result.n_inliers)
        {
            best_result = result;
        }
    }

    std::cout << best_result.n_inliers << std::endl;
    return best_result.plane;
}
