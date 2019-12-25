#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import kmeans_pytorch


class UnitTests(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(kmeans_pytorch)

    def test_project(self):
        self.assertTrue(False, "write more tests here")