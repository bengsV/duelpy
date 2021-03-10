#######
Release
#######

It's time. Let's make a release. Don't rush it though. It's easy to miss
something.

Essential
---------

These are the essential steps. Make sure to follow them before releasing.

- Check the `changelog`_.
    - Make sure it is complete. Add entries if necessary.
    - Check whether or not there were any breaking changes. Decide on the next
      version number according to Semantic Versioning. If there are breaking
      changes and you still want to create a point release, use a release
      branch without the breaking changes.
- Make a commit for the release.
    - Update the version in ``setup.py``.
    - Update the version in the `changelog`_.
        - Rename the *Unreleased* section to the new version. Create a new
          *Unreleased* section. Define a link reference for the new version at
          the bottom of the file. Follow the existing format. The link target
          will not be valid yet at this point, since the tag does not exist.
    - Check for any remaining mentions of the previous version
      (``git grep -F 'X.Y.Z'``) and update them as necessary. Add an entry here
      if you find a new place that needs to be updated.
- Create a Merge Request for the commit, titled "Release X.Y.Z". Copy this
  checklist into the description:

.. code-block:: Markdown

    [ ] The new version follows SemVer.
    [ ] The Changelog is up to date.
    [ ] The Changelog has a section for the new version.
    [ ] The Changelog has a link for the new version.
    [ ] The version is updated in `setup.py`.
    [ ] I have checked for other mentions of the old version.

- Double check each of the entries. Mark them as done when they are done.
- Wait for a review. Merge the MR.
- Create a new tag with the format ``vX.Y.Z`` that points to the merge commit.
- The "pypi" job of the GitLab `pipeline`_ should push the release to `PyPI`_. You
  can check its status `here`_.

.. _pipeline: https://gitlab.com/duelpy/duelpy/-/blob/master/.gitlab-ci.yml
.. _PyPI: https://pypi.org/project/duelpy/
.. _here: https://gitlab.com/duelpy/duelpy/-/pipelines

Polish
------

Here are some quality-control points which are nice to have, but can always be
fixed in a follow-up release if necessary.

- Check the `documentation`_.
    - Are there any jarring formatting errors? Does the documentation render
      correctly? You do not need to check every page, but skim at least
      a couple of them. The `changelog`_ can serve as a guide here.
- Check the `README`_.
    - Is it still accurate and up to date?

.. _changelog: https://gitlab.com/duelpy/duelpy/-/blob/master/CHANGELOG.md
.. _documentation: https://duelpy.gitlab.io/duelpy/
.. _README: https://gitlab.com/duelpy/duelpy/-/blob/master/README.md
