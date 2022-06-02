from setuptools import setup

setup(
    name='value_at_risk',
    version='1.0.1',
    author='moodoid',
    keywords='Value-at-Risk Tool',
    packages=['value_at_risk'],
    description='Value at Risk Calculator',
    long_description='Calculate Value-at-Risk (VaR) of a portfolio through historical and parametric methods',
    long_description_content_type='text/plain',
    url='https://github.com/moodoid/value_at_risk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'LICENSE :: OSI Approved :: GNU Lesser General Public LICENSE v3 (LGPLv3)',
        'Programming Language :: Python :: 3'
    ]
)