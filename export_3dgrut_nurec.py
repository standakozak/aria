"""Export a 3DGRUT checkpoint to USD/NuRec with a working ``--no-transform`` flag.

Thin wrapper around ``threedgrut.export.scripts.export_usd`` (same arguments).
In 3DGRUT, ``--no-transform`` is only passed to the standard USD exporter; the
NuRec exporter instead reads ``export_usd.apply_normalizing_transform`` from the
config stored in the checkpoint (``true`` by default). This wrapper sets that
value to ``false`` after loading the checkpoint when ``--no-transform`` is given,
so the Gaussians stay in the COLMAP world frame.

Why: with the normalizing transform, the NuRec exporter writes the transform in
column-vector order (translation in the last column), which USD interprets as a
projective transform. ``ply_to_usd.py --gaussian-usdz`` copies this transform onto
the mesh, which distorts it.

Run with the "aria" environment from the 3DGRUT directory:
    python ../aria/export_3dgrut_nurec.py -c runs/<experiment>/<run>/ckpt_last.pt \
        -o runs/<experiment>/<run>/export_nurec_world.usdz --format nurec --no-transform
"""

import sys

import threedgrut.export.scripts.export_usd as export_usd

if "--no-transform" in sys.argv:
    _load_model_from_checkpoint = export_usd.load_model_from_checkpoint

    def _load_without_normalizing_transform(checkpoint_path):
        model, conf, background, post_processing = _load_model_from_checkpoint(checkpoint_path)
        conf.export_usd.apply_normalizing_transform = False
        return model, conf, background, post_processing

    export_usd.load_model_from_checkpoint = _load_without_normalizing_transform

if __name__ == "__main__":
    sys.exit(export_usd.main())
