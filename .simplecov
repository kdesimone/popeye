require 'simplecov'
require 'coveralls'

SimpleCov.formatter = Coveralls::SimpleCov::Formatter
SimpleCov.start do
    add filter 'popeye/plotting.py'
    add filter 'popeye/reconstruction.py'
    add filter 'popeye/simulation.py'
    add filter 'popeye/spectrotemporal.py'
    add filter 'popeye/spatiotemporal*py'
    add filter 'popeye/xvalidation.py'
    add filter 'popeye/strf.py'
end

